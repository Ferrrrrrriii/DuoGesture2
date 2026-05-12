"""Generate a findings summary for a training run.

Writes `findings.txt` into the run directory with:
- training-config YAML snippet
- extracted physics-related metrics from the training log
- scores extracted from the training log (FID / BC / L1div)
- fresh FGD re-run using the VAESKConv encoder
- comparison with other 'sparse' runs (if available)

Usage:
    python utils/write_findings.py --run_dir outputs/custom/<run> [--epoch 45] [--device cpu]

"""
import os
import sys
import glob
import argparse
import textwrap
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.run_fgd_eval import (
    load_eval_encoder,
    compute_fgd_from_dir,
    extract_scores_from_log,
    _find_log_file,
)


def read_yaml_snippet(run_dir: str, max_lines: int = 200) -> str:
    yfiles = glob.glob(os.path.join(run_dir, "*.yaml"))
    if not yfiles:
        return "(no yaml config found)"
    p = yfiles[0]
    try:
        with open(p, "r") as f:
            lines = f.readlines()[:max_lines]
        return "".join(lines)
    except Exception as e:
        return f"(failed to read {p}: {e})"


def extract_physics_lines(log_path: str, n: int = 10) -> str:
    if not log_path or not os.path.isfile(log_path):
        return "(no log file)"
    lines = open(log_path, "r").read().splitlines()
    phys_lines = [l for l in lines if "phys_" in l or "phys" in l]
    if not phys_lines:
        # fallback: return the last n lines of the log
        return "\n".join(lines[-n:])
    return "\n".join(phys_lines[-n:])


def find_sparse_runs(root_outputs: str) -> list:
    # find run dirs under outputs/custom that contain 'sparse' in their name
    runs = []
    custom = os.path.join(root_outputs, "custom")
    if not os.path.isdir(custom):
        return runs
    for d in os.listdir(custom):
        if "sparse" in d:
            runs.append(os.path.join(custom, d))
    return runs


def compare_with_sparse(run_epoch_dir: str, encoder, device: str) -> list:
    # For each sparse run, try to compute FGD on the same epoch name if exists
    results = []
    sparse_runs = find_sparse_runs(os.path.join(ROOT, "outputs"))
    epoch_name = os.path.basename(run_epoch_dir.rstrip("/"))
    for srun in sparse_runs:
        candidate = os.path.join(srun, epoch_name)
        if os.path.isdir(candidate):
            try:
                res = compute_fgd_from_dir(candidate, encoder, device)
                if "error" in res:
                    results.append((srun, None, res.get("error")))
                else:
                    results.append((srun, float(res["fgd"]), None))
            except Exception as e:
                results.append((srun, None, str(e)))
    return results


def make_findings(run_dir: str, epoch: int, eval_ckpt: str, device: str) -> str:
    # Resolve epoch dir
    epoch_dir = os.path.join(run_dir, str(epoch)) if epoch is not None else None

    # 1) config snippet
    cfg_snip = read_yaml_snippet(run_dir)

    # 2) find log file and extract scores + physics
    log_file = _find_log_file(run_dir)
    log_scores = {}
    physics_snip = "(no log found)"
    if log_file and epoch is not None:
        log_scores = extract_scores_from_log(log_file, epoch)
        physics_snip = extract_physics_lines(log_file)
    else:
        if log_file:
            physics_snip = extract_physics_lines(log_file)

    # 3) Load encoder and compute fresh FGD if epoch_dir exists
    encoder = load_eval_encoder(eval_ckpt, device)
    fgd_res = None
    if epoch_dir and os.path.isdir(epoch_dir):
        fgd_res = compute_fgd_from_dir(epoch_dir, encoder, device)

    # 4) compare with sparse runs
    sparse_cmp = []
    if epoch_dir and fgd_res and "error" not in fgd_res:
        sparse_cmp = compare_with_sparse(epoch_dir, encoder, device)

    # 5) compose findings text
    parts = []
    parts.append("=== FINDINGS SUMMARY ===")
    parts.append(f"Run dir : {run_dir}")
    parts.append(f"Epoch   : {epoch}")
    parts.append("")
    parts.append("-- Config snippet --")
    parts.append(cfg_snip)
    parts.append("")
    parts.append("-- Physics / training-specific metrics (excerpt from log) --")
    parts.append(physics_snip)
    parts.append("")
    parts.append("-- Scores extracted from training log (if available) --")
    if log_scores:
        for k, v in log_scores.items():
            parts.append(f"{k}: {v}")
    else:
        parts.append("(no scores parsed from log)")

    parts.append("")
    parts.append("-- Fresh FGD re-run (VAESKConv encoder) --")
    if fgd_res is None:
        parts.append("(epoch dir missing or FGD re-run not available)")
    elif "error" in fgd_res:
        parts.append(f"FGD error: {fgd_res['error']}")
    else:
        parts.append(f"FGD: {fgd_res['fgd']:.6f}")
        parts.append(f"n_seqs: {fgd_res['n_seqs']}")
        parts.append(f"gt_shape: {fgd_res['gt_shape']}  res_shape: {fgd_res['res_shape']}")

    parts.append("")
    parts.append("-- Comparison with sparse runs (same epoch name, if present) --")
    if not sparse_cmp:
        parts.append("(no sparse runs with matching epoch found)")
    else:
        for srun, fgd_val, err in sparse_cmp:
            if err:
                parts.append(f"{srun} : ERROR {err}")
            else:
                parts.append(f"{srun} : FGD = {fgd_val:.6f}")

    parts.append("")
    parts.append("-- How FGD is computed (brief) --")
    parts.append(textwrap.dedent(
        """
        FGD (Fréchet Gesture Distance) compares Gaussian statistics of latent
        features extracted from GT and generated gestures. Steps:
         1) Extract frame/windows features via an encoder (RVQVAE or VAESKConv).
         2) Compute empirical mean (μ) and covariance (Σ) over feature windows.
         3) FGD = ||μ_r - μ_g||^2 + Tr(Σ_r + Σ_g - 2*(Σ_r Σ_g)^(1/2)).

        In this project we typically use a VAESKConv encoder for the re-run
        FGD (latent dim 240) and RVQVAE per-body-part encoders for part-wise
        FGD. The script writes both the fresh FGD and references to the
        training-log FID/BC/L1div when available.
        """))

    parts.append("")
    parts.append("-- Notes about embeddings & physics findings --")
    parts.append(textwrap.dedent(
        """
        - Audio embeddings: training uses HuBERT-style audio features (see
          'hubert_cons' entries in the log) and optionally MoCLIP conditioning
          depending on config flags in the YAML snippet above.
        - Embedding pipeline: audio -> HuBERT features -> encoder -> latent
          representations used by the motion generator. The 'latent' and
          'latent_word' losses in the log quantify alignment between audio
          semantics and generated motion.
        - Physics terms: 'phys_jerk', 'phys_beta' and other 'phys_*' metrics
          reflect the physics-regulariser and its effect on motion dynamics.
        - To attribute an FGD result to a combination (e.g., base + sparse +
          physics), this summary reports both the log-extracted FID (includes
          SMPL-X BC) and the fresh FGD re-run that isolates latent-space
          statistical divergence.
        """))

    parts.append("")
    parts.append("-- End of findings --")

    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser(description="Write findings summary for a run")
    p.add_argument("--run_dir", required=True, help="Path to run directory (e.g. outputs/custom/...) ")
    p.add_argument("--epoch", type=int, default=45, help="Epoch number folder to evaluate (default: 45)")
    p.add_argument("--eval_ckpt", default="BEAT2/beat_english_v2.0.0/weights/AESKConv_240_100.bin",
                   help="Eval encoder checkpoint path (relative to project root)")
    p.add_argument("--device", default="cpu", help="Torch device to run encoder on (cpu/cuda)")
    args = p.parse_args()

    run_dir = args.run_dir
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(ROOT, run_dir)

    findings = make_findings(run_dir, args.epoch, args.eval_ckpt, args.device)

    out_path = os.path.join(run_dir, "findings.txt")
    with open(out_path, "a") as f:
        f.write("\n" + findings + "\n")

    # also append a short marker to the training .txt log if present
    log_file = _find_log_file(run_dir)
    marker = f"\n[INFO] findings written to {out_path}\n"
    if log_file:
        try:
            with open(log_file, "a") as lf:
                lf.write(marker)
        except Exception:
            pass

    print(f"[DONE] findings written to: {out_path}")


if __name__ == "__main__":
    main()
