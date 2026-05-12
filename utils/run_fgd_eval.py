"""
FGD + BC + L1div evaluation for DuoGesture — single-run and sweep-comparison modes.

Evaluation protocol (matches duogesture_sparse_trainer.py test loop exactly):
  FGD  — Fréchet Gesture Distance using VAESKConv encoder (AESKConv_240_100.bin,
          latent_dim=240). Uses gt_*.npz / res_*.npz written to the epoch dir
          during training. Identical to the inline 'fid score:' in training logs.
  BC   — Beat Consistency (beat alignment score). Extracted from the training log
          at the best epoch — computed inside the trainer using librosa onset
          detection + SMPL-X 3D joint velocity, matching the 'align score:' lines.
  L1div — Pose diversity (L1 distance from mean in joint-angle space). Extracted
          from the training log — matches the 'l1div score:' lines.

⚠ Scope caveat:
  All three metrics cover the 15 TEST sequences of speaker 2 (Scott) only.
  The full BEAT2 English test split contains 265 sequences across 25 speakers.
  The model was trained on Scott only (training_speakers: [2]), so single-speaker
  evaluation is internally consistent but NOT directly comparable to published
  multi-speaker results (CaMN, DiffGesture, etc.).

Usage
-----
Single epoch dir (re-run FGD fresh):
    python utils/run_fgd_eval.py --epoch_dir outputs/custom/<run>/<epoch>

Sweep comparison — reads summary.csv, evaluates all TRAINED runs, prints table,
writes FID/BC/L1div back into the CSV:
    python utils/run_fgd_eval.py \\
        --sweep_csv outputs/sweeps/<name>/summary.csv \\
        [--device cuda] [--no_csv_update]
"""

import sys, os, re, csv, argparse, glob, types, copy
from typing import Optional, List, Dict, Any
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d
from utils.fgd import compute_statistics, calculate_fgd
from dataloaders.data_tools import FIDCalculator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_EPOCH_DIR = "outputs/custom/0216_084918_duogesture_moclip_sparse/138"
EVAL_CKPT         = "BEAT2/beat_english_v2.0.0/weights/AESKConv_240_100.bin"
DATA_PATH_1       = "./weights/"
VAE_TEST_LEN      = 32    # chunk size (matches duogesture_sparse.yaml vae_test_len)
VAE_LENGTH        = 240   # latent dim  (matches AESKConv_240_100)
VAE_TEST_DIM      = 330   # 55 joints × 6 rot6d
N_JOINTS          = 55

# Metrics subset label written into summary.csv
SUBSET_LABEL = "scott_15"   # 15 test sequences from speaker 2 / Scott


# ---------------------------------------------------------------------------
# VAESKConv helpers
# ---------------------------------------------------------------------------
def _make_eval_args(data_path_1: str = DATA_PATH_1) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        vae_test_dim     = VAE_TEST_DIM,
        vae_test_len     = VAE_TEST_LEN,
        vae_length       = VAE_LENGTH,
        vae_codebook_size= 256,
        vae_layer        = 4,
        vae_grow         = [1, 1, 2, 1],
        variational      = False,
        data_path_1      = data_path_1,
    )


def load_eval_encoder(ckpt_path: str, device: str = "cpu") -> nn.Module:
    from models.motion_representation import VAESKConv
    args  = _make_eval_args()
    model = VAESKConv(args).to(device).eval()
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Eval encoder checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state.get("model_state", state)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    print(f"[INFO] Loaded eval encoder: {os.path.basename(ckpt_path)}")
    return model


def poses_to_rot6d(poses_aa: np.ndarray) -> torch.Tensor:
    """(N, 165) axis-angle → (N, 330) rot6d — matches trainer preprocessing."""
    t   = torch.from_numpy(poses_aa).float()
    N   = t.shape[0]
    aa  = t.reshape(N, N_JOINTS, 3)
    mat = axis_angle_to_matrix(aa)
    r6d = matrix_to_rotation_6d(mat)
    return r6d.reshape(N, N_JOINTS * 6)


def extract_latents(npz_paths: list, encoder: nn.Module, device: str = "cpu") -> np.ndarray:
    """
    Load NPZ files, convert poses to rot6d, chunk into VAE_TEST_LEN windows,
    encode, return (M_total, VAE_LENGTH) float64 array.
    Matches duogesture_sparse_trainer.py test loop latent extraction exactly.
    """
    latents    = []
    enc_device = next(encoder.parameters()).device
    for p in sorted(npz_paths):
        data   = np.load(p, allow_pickle=True)
        poses  = data["poses"]              # (N_frames, 165)
        r6d    = poses_to_rot6d(poses)      # (N_frames, 330)
        n      = r6d.shape[0]
        n_trim = n - (n % VAE_TEST_LEN)
        if n_trim == 0:
            continue
        r6d_trim = r6d[:n_trim].unsqueeze(0).to(enc_device)
        with torch.no_grad():
            z = encoder.map2latent(r6d_trim)
            z = z.reshape(-1, VAE_LENGTH)
        latents.append(z.cpu().numpy())
    return np.concatenate(latents, axis=0)


def compute_fgd_from_dir(epoch_dir: str, encoder: nn.Module, device: str) -> dict:
    """
    Run fresh FGD on gt_*.npz / res_*.npz in epoch_dir.
    Returns dict with keys: fgd, n_seqs, gt_shape, res_shape.
    """
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(epoch_dir):
        epoch_dir = os.path.join(ROOT, epoch_dir)

    gt_paths  = sorted(glob.glob(os.path.join(epoch_dir, "gt_*.npz")))
    res_paths = sorted(glob.glob(os.path.join(epoch_dir, "res_*.npz")))

    if not gt_paths:
        return {"error": f"No gt_*.npz in {epoch_dir}"}
    if len(gt_paths) != len(res_paths):
        return {"error": f"GT/res mismatch: {len(gt_paths)} vs {len(res_paths)}"}

    gt_lat  = extract_latents(gt_paths,  encoder, device)
    res_lat = extract_latents(res_paths, encoder, device)

    fgd = FIDCalculator.frechet_distance(res_lat, gt_lat)
    return {
        "fgd": float(fgd),
        "n_seqs": len(gt_paths),
        "gt_shape": gt_lat.shape,
        "res_shape": res_lat.shape,
    }


# ---------------------------------------------------------------------------
# Training-log extraction  (FID / BC / L1div at a specific epoch)
# ---------------------------------------------------------------------------
def extract_scores_from_log(log_path: str, best_epoch: int) -> dict:
    """
    Parse a DuoGesture training .txt log to find the FID (inline fid score:),
    BC (align score:), and L1div (l1div score:) that were logged right after
    the best_epoch training step.

    Returns dict with keys fid, bc, l1div (floats) or error string.
    """
    if not os.path.isfile(log_path):
        return {"error": f"Log not found: {log_path}"}

    lines = open(log_path).readlines()

    # Find the first line of the best_epoch training step [best_epoch][000/...]
    epoch_line = None
    for i, l in enumerate(lines):
        if re.search(r'\[' + str(best_epoch) + r'\]\[0+0/', l):
            epoch_line = i
            break

    # Fallback: last training line belonging to best_epoch
    if epoch_line is None:
        for i in range(len(lines) - 1, -1, -1):
            if re.search(r'\[' + str(best_epoch) + r'\]\[', lines[i]):
                epoch_line = i
                break

    if epoch_line is None:
        return {"error": f"Epoch {best_epoch} not found in log"}

    # Scan forward for fid/align/l1div score lines (logged right after training)
    fid = bc = l1div = None
    for j in range(epoch_line, min(epoch_line + 30, len(lines))):
        l = lines[j]
        if fid    is None and "fid score:"   in l:
            m = re.search(r'fid score:\s*([0-9.eE+\-]+)', l)
            if m: fid = float(m.group(1))
        if bc     is None and "align score:" in l:
            m = re.search(r'align score:\s*([0-9.eE+\-]+)', l)
            if m: bc = float(m.group(1))
        if l1div  is None and "l1div score:" in l:
            m = re.search(r'l1div score:\s*([0-9.eE+\-]+)', l)
            if m: l1div = float(m.group(1))
        if fid is not None and bc is not None and l1div is not None:
            break

    result = {}
    if fid    is None: result["warn_fid"]   = f"fid not found after epoch {best_epoch}"
    else:              result["fid"]        = fid
    if bc     is None: result["warn_bc"]    = f"align not found after epoch {best_epoch}"
    else:              result["bc"]         = bc
    if l1div  is None: result["warn_l1div"] = f"l1div not found after epoch {best_epoch}"
    else:              result["l1div"]      = l1div
    return result


def _find_log_file(run_dir: str) -> Optional[str]:
    """Find the .txt training log inside a run directory."""
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(ROOT, run_dir)
    candidates = sorted(glob.glob(os.path.join(run_dir, "*.txt")))
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def read_sweep_csv(csv_path: str) -> List[Dict[str, Any]]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def write_sweep_csv(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Single-run mode
# ---------------------------------------------------------------------------
def run_single(epoch_dir: str, eval_ckpt: str, device: str) -> None:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(eval_ckpt):
        eval_ckpt = os.path.join(ROOT, eval_ckpt)

    encoder = load_eval_encoder(eval_ckpt, device)
    result  = compute_fgd_from_dir(epoch_dir, encoder, device)

    if "error" in result:
        sys.exit(f"[ERROR] {result['error']}")

    print(f"\n{'='*62}")
    print(f"  FGD Evaluation — DuoGesture MoCLIP")
    print(f"{'='*62}")
    print(f"  Epoch dir : {epoch_dir}")
    print(f"  Sequences : {result['n_seqs']}  (Scott test = 15 / full BEAT2 test = 265)")
    print(f"  Encoder   : VAESKConv  ({os.path.basename(eval_ckpt)})")
    print(f"  GT latents: {result['gt_shape']}   Gen latents: {result['res_shape']}")
    print(f"  Device    : {device}")
    print(f"{'='*62}")
    print(f"  FGD  : {result['fgd']:.4f}")
    print(f"  BC / L1div: see training log (need SMPL-X forward pass)")
    print(f"\n  ⚠  Single-speaker results (Scott only). Not paper-comparable.")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# Sweep-comparison mode
# ---------------------------------------------------------------------------
def run_sweep(csv_path: str, eval_ckpt: str, device: str, update_csv: bool) -> None:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(eval_ckpt):
        eval_ckpt = os.path.join(ROOT, eval_ckpt)
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(ROOT, csv_path)

    rows = read_sweep_csv(csv_path)
    trained = [r for r in rows if (r.get("status") or "").strip().startswith("TRAINED")]

    if not trained:
        sys.exit("[ERROR] No TRAINED rows found in summary.csv")

    print(f"\n{'='*72}")
    print(f"  Sweep FGD / BC / L1div comparison")
    print(f"  CSV   : {csv_path}")
    print(f"  Runs  : {len(trained)} TRAINED  (full sweep has {len(rows)} rows)")
    print(f"  Subset: {SUBSET_LABEL} — 15 test sequences, speaker 2 (Scott)")
    print(f"  ⚠  Not directly comparable to published multi-speaker BEAT2 results")
    print(f"{'='*72}\n")

    encoder = load_eval_encoder(eval_ckpt, device)

    results = []   # list of dicts for table
    csv_updates = {}   # run_id → {fgd_subset, fid, bc, l1div}

    for row in trained:
        run_id    = row["run_id"]
        run_dir   = row["run_dir"]
        best_ckpt = row.get("best_ckpt", "")
        try:
            best_epoch = int(row.get("best_epoch", 0))
        except (ValueError, TypeError):
            best_epoch = 0

        print(f"  [{run_id}]  best_epoch={best_epoch}")

        # ── 1. Extract FID / BC / L1div from training log ─────────────────
        log_file = _find_log_file(run_dir)
        log_scores = {}
        if log_file and best_epoch > 0:
            log_scores = extract_scores_from_log(log_file, best_epoch)
            for k, v in log_scores.items():
                if k.startswith("warn_") or k == "error":
                    print(f"         ⚠  {v}")
        else:
            log_scores["error"] = f"no log file found in {run_dir}"
            print(f"         ⚠  {log_scores['error']}")

        # ── 2. Re-run FGD on epoch_dir NPZs (cross-verification) ──────────
        if not os.path.isabs(run_dir):
            abs_run_dir = os.path.join(ROOT, run_dir)
        else:
            abs_run_dir = run_dir
        epoch_dir = os.path.join(abs_run_dir, str(best_epoch))

        fgd_fresh = None
        n_seqs    = 0
        if os.path.isdir(epoch_dir):
            fgd_res = compute_fgd_from_dir(epoch_dir, encoder, device)
            if "error" in fgd_res:
                print(f"         ⚠  FGD re-run failed: {fgd_res['error']}")
            else:
                fgd_fresh = fgd_res["fgd"]
                n_seqs    = fgd_res["n_seqs"]
                print(f"         FGD (re-run)  : {fgd_fresh:.4f}  ({n_seqs} seqs)")
        else:
            print(f"         ⚠  epoch dir not found: {epoch_dir}")

        # ── 3. Consolidate: prefer log-extracted scores (include SMPL-X BC) ─
        fid_final   = log_scores.get("fid",   fgd_fresh)
        bc_final    = log_scores.get("bc",    None)
        l1div_final = log_scores.get("l1div", None)

        if fid_final is not None:
            print(f"         FID (log)    : {fid_final:.4f}")
        if bc_final is not None:
            print(f"         BC  (log)    : {bc_final:.4f}")
        if l1div_final is not None:
            print(f"         L1div (log)  : {l1div_final:.4f}")
        print()

        results.append({
            "run_id":      run_id,
            "best_epoch":  best_epoch,
            "fid":         fid_final,
            "bc":          bc_final,
            "l1div":       l1div_final,
            "fgd_rerun":   fgd_fresh,
            "n_seqs":      n_seqs,
        })

        if update_csv:
            csv_updates[run_id] = {
                "fgd_subset": SUBSET_LABEL,
                "fid":        f"{fid_final:.6f}"   if fid_final   is not None else "",
                "bc":         f"{bc_final:.6f}"    if bc_final    is not None else "",
                "l1div":      f"{l1div_final:.6f}" if l1div_final is not None else "",
            }

    # ── Comparison table ──────────────────────────────────────────────────
    valid = [r for r in results if r["fid"] is not None]
    if valid:
        ranked = sorted(valid, key=lambda r: r["fid"])
        print(f"\n{'='*72}")
        print(f"  {'Run':<22}  {'Epoch':>5}  {'FID↓':>8}  {'BC↑':>8}  {'L1div':>8}  {'FGD re-run':>10}")
        print(f"  {'-'*22}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
        for r in ranked:
            fid_str    = f"{r['fid']:.4f}"    if r["fid"]    is not None else "   N/A"
            bc_str     = f"{r['bc']:.4f}"     if r["bc"]     is not None else "   N/A"
            l1div_str  = f"{r['l1div']:.3f}"  if r["l1div"]  is not None else "  N/A"
            rerun_str  = f"{r['fgd_rerun']:.4f}" if r["fgd_rerun"] is not None else "     N/A"
            marker = " ←best" if r is ranked[0] else ""
            print(f"  {r['run_id']:<22}  {r['best_epoch']:>5}  {fid_str:>8}  {bc_str:>8}  {l1div_str:>8}  {rerun_str:>10}{marker}")
        print(f"{'='*72}")
        print(f"  FID ↓ lower is better  |  BC ↑ higher is better  |  L1div ↑ higher is better")
        print(f"  Subset: {SUBSET_LABEL} — 15 Scott test seqs. NOT paper-comparable.")
        print(f"{'='*72}\n")

    # ── Update CSV ────────────────────────────────────────────────────────
    if update_csv and csv_updates:
        updated_rows = []
        for row in rows:
            new_row = copy.copy(row)
            if row["run_id"] in csv_updates:
                new_row.update(csv_updates[row["run_id"]])
            updated_rows.append(new_row)
        write_sweep_csv(csv_path, updated_rows)
        print(f"[INFO] Updated {csv_path} with FID/BC/L1div for {len(csv_updates)} runs.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="FGD + BC + L1div evaluation for DuoGesture (single-run or sweep)."
    )
    # Modes (mutually exclusive)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--epoch_dir",  default=None,
                      help="Single epoch dir with gt_*.npz + res_*.npz")
    mode.add_argument("--sweep_csv",  default=None,
                      help="Path to sweep summary.csv — evaluates ALL TRAINED runs and prints comparison table")

    parser.add_argument("--eval_ckpt", default=EVAL_CKPT,
                        help="AESKConv_240_100.bin checkpoint (default: BEAT2/...)")
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_csv_update", action="store_true",
                        help="Do NOT write FID/BC/L1div back into summary.csv (sweep mode only)")

    args = parser.parse_args()

    if args.sweep_csv:
        run_sweep(args.sweep_csv, args.eval_ckpt, args.device,
                  update_csv=not args.no_csv_update)
    else:
        epoch_dir = args.epoch_dir or DEFAULT_EPOCH_DIR
        run_single(epoch_dir, args.eval_ckpt, args.device)


if __name__ == "__main__":
    main()

