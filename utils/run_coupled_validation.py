import argparse
import glob
import json
import os

import numpy as np
from scipy.signal import hilbert, welch


DEFAULT_SHOULDER_DEFS = {
    "shoulders_only": [16, 17],
    "shoulder_elbow": [16, 17, 18, 19],
    "upper_limb": [16, 17, 18, 19, 20, 21],
}

DEFAULT_CORE_DEFS = {
    "torso_4": [0, 3, 6, 9],
    "torso_plus_neck": [0, 3, 6, 9, 12],
    "pelvis_spine": [0, 3, 6],
}

FOREARM_IDX = [18, 19, 20, 21]
HAND_IDX = [20, 21]


def _safe_norm(x, axis=-1, keepdims=False, eps=1e-8):
    n = np.linalg.norm(x, axis=axis, keepdims=keepdims)
    if np.isscalar(n):
        return max(n, eps)
    return np.maximum(n, eps)


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    n = min(len(y_true), len(y_pred))
    if n < 2:
        return 0.0
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _fit_linear(X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim == 1:
        X = X[:, None]
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    w, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
    y_hat = Xb @ w
    return w, y_hat


def _speaker_id_from_name(path):
    token = os.path.basename(path).split("_")[0]
    try:
        return int(token)
    except Exception:
        return token


def _condition_from_name(path):
    name = os.path.basename(path).lower()
    if "gt" in name and (name.startswith("gt_") or "_gt_" in name):
        return "gt"
    if "moclip" in name:
        return "moclip"
    if "svib" in name:
        return "svib"
    if "base" in name:
        return "base"
    return "other"


def _load_npz_pose(path):
    with np.load(path, allow_pickle=True) as arr:
        if "poses" in arr:
            poses = np.asarray(arr["poses"])
        elif "pose" in arr:
            poses = np.asarray(arr["pose"])
        elif "joints" in arr:
            poses = np.asarray(arr["joints"])
        else:
            return None, None

        fps = float(np.asarray(arr["mocap_frame_rate"]).reshape(-1)[0]) if "mocap_frame_rate" in arr else 30.0

    if poses.ndim == 2 and poses.shape[-1] % 3 == 0:
        j = poses.shape[-1] // 3
        poses = poses.reshape(poses.shape[0], j, 3)
    if poses.ndim != 3 or poses.shape[-1] != 3:
        return None, None
    if poses.shape[1] < 22:
        return None, None
    return poses.astype(np.float64), fps


def _angular_velocity_signal(poses, joint_indices, fps):
    if poses.shape[0] < 4:
        return None
    idx = [i for i in joint_indices if i < poses.shape[1]]
    if len(idx) == 0:
        return None
    omega = np.diff(poses[:, idx, :], axis=0) * float(fps)
    flat = omega.reshape(omega.shape[0], -1)
    flat = flat - np.mean(flat, axis=0, keepdims=True)

    # Use signed dominant direction to avoid frequency folding from ||omega||.
    if flat.shape[1] == 1:
        return flat[:, 0]
    try:
        _, _, vh = np.linalg.svd(flat, full_matrices=False)
        axis = vh[0]
        sig = flat @ axis
        if np.std(sig) < 1e-10:
            return np.mean(flat, axis=1)
        return sig
    except Exception:
        return np.mean(flat, axis=1)


def _welch_peak_hz(sig, fs, f_min=0.3, f_max=8.0):
    sig = np.asarray(sig).reshape(-1)
    n = len(sig)
    if n < 12:
        return np.nan, np.nan
    sig = sig - np.mean(sig)
    nperseg = int(min(128, n))
    noverlap = int(max(0, nperseg // 2))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="linear", window="hann")
    band = (freqs >= f_min) & (freqs <= f_max)
    if np.sum(band) < 2:
        return np.nan, np.nan
    fb = freqs[band]
    pb = psd[band]
    peak_idx = int(np.argmax(pb))
    return float(fb[peak_idx]), float(pb[peak_idx])


def _phase_metrics(x, y, fps, max_lag_s=0.6):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    n = min(len(x), len(y))
    if n < 8:
        return 0.0, 0, 0.0, 0.0
    x = x[:n] - np.mean(x[:n])
    y = y[:n] - np.mean(y[:n])

    phx = np.angle(hilbert(x))
    phy = np.angle(hilbert(y))
    plv = float(np.abs(np.mean(np.exp(1j * (phx - phy)))))

    max_lag = int(max(1, round(max_lag_s * fps)))
    xc = np.correlate(x, y, mode="full")
    lags = np.arange(-n + 1, n)
    mask = (lags >= -max_lag) & (lags <= max_lag)
    xc_sub = xc[mask]
    lags_sub = lags[mask]
    best = int(np.argmax(xc_sub))
    lag_frames = int(lags_sub[best])
    lag_seconds = float(lag_frames / float(fps))
    lag_corr = float(xc_sub[best] / max(_safe_norm(x), 1e-12) / max(_safe_norm(y), 1e-12))
    return plv, lag_frames, lag_seconds, lag_corr


def _oscillator_fit(x1, x2, dt):
    x1 = np.asarray(x1).reshape(-1)
    x2 = np.asarray(x2).reshape(-1)
    n = min(len(x1), len(x2))
    if n < 10:
        return {
            "r2_upper": 0.0,
            "r2_forearm": 0.0,
            "k_upper_to_forearm": 0.0,
            "k_forearm_to_upper": 0.0,
            "coupling_asymmetry": 0.0,
        }
    x1 = x1[:n]
    x2 = x2[:n]
    dx1 = np.gradient(x1, dt)
    dx2 = np.gradient(x2, dt)
    ddx1 = np.gradient(dx1, dt)
    ddx2 = np.gradient(dx2, dt)

    X1 = np.stack([dx1, x1, (x1 - x2)], axis=1)
    y1 = -ddx1
    w1, y1_hat = _fit_linear(X1, y1)

    X2 = np.stack([dx2, x2, (x2 - x1)], axis=1)
    y2 = -ddx2
    w2, y2_hat = _fit_linear(X2, y2)

    k12 = float(w1[2])
    k21 = float(w2[2])
    return {
        "r2_upper": float(_r2(y1, y1_hat)),
        "r2_forearm": float(_r2(y2, y2_hat)),
        "k_upper_to_forearm": k12,
        "k_forearm_to_upper": k21,
        "coupling_asymmetry": float(abs(k12 - k21)),
    }


def _make_lag_features(signals_dict, keys, max_lag):
    arrays = [np.asarray(signals_dict[k]).reshape(-1) for k in keys if k in signals_dict and signals_dict[k] is not None]
    if len(arrays) == 0:
        return None, None
    n = min([len(a) for a in arrays])
    if n <= max_lag + 2:
        return None, None
    feats = []
    for k in keys:
        if k not in signals_dict or signals_dict[k] is None:
            continue
        s = np.asarray(signals_dict[k]).reshape(-1)[:n]
        for lag in range(max_lag + 1):
            feats.append(s[max_lag - lag:n - lag])
    X = np.stack(feats, axis=1)
    y = np.asarray(signals_dict["hand"]).reshape(-1)[:n][max_lag:]
    return X, y


def _predictor_delta_r2(shoulder, forearm, core, hand, max_lag=4):
    seq = {
        "shoulder": np.asarray(shoulder).reshape(-1),
        "forearm": np.asarray(forearm).reshape(-1),
        "core": np.asarray(core).reshape(-1),
        "hand": np.asarray(hand).reshape(-1),
    }
    n = min([len(v) for v in seq.values()])
    if n < max_lag + 24:
        return 0.0
    for k in seq.keys():
        seq[k] = seq[k][:n]

    split = int(0.7 * n)
    if split <= max_lag + 5 or (n - split) <= max_lag + 5:
        return 0.0

    tr = {k: v[:split] for k, v in seq.items()}
    te = {k: v[split:] for k, v in seq.items()}

    Xtr_s, ytr_s = _make_lag_features(tr, ["shoulder"], max_lag)
    Xte_s, yte_s = _make_lag_features(te, ["shoulder"], max_lag)
    Xtr_full, ytr_f = _make_lag_features(tr, ["shoulder", "forearm", "core"], max_lag)
    Xte_full, yte_f = _make_lag_features(te, ["shoulder", "forearm", "core"], max_lag)

    if Xtr_s is None or Xte_s is None or Xtr_full is None or Xte_full is None:
        return 0.0

    ws, _ = _fit_linear(Xtr_s, ytr_s)
    wf, _ = _fit_linear(Xtr_full, ytr_f)

    Xte_s_b = np.concatenate([Xte_s, np.ones((Xte_s.shape[0], 1), dtype=np.float64)], axis=1)
    Xte_f_b = np.concatenate([Xte_full, np.ones((Xte_full.shape[0], 1), dtype=np.float64)], axis=1)
    ys_hat = Xte_s_b @ ws
    yf_hat = Xte_f_b @ wf

    r2_s = _r2(yte_s, ys_hat)
    r2_f = _r2(yte_f, yf_hat)
    return float(r2_f - r2_s)


def _sequence_metrics(poses, fps, shoulder_idx, core_idx):
    shoulder = _angular_velocity_signal(poses, shoulder_idx, fps)
    forearm = _angular_velocity_signal(poses, FOREARM_IDX, fps)
    core = _angular_velocity_signal(poses, core_idx, fps)
    hand = _angular_velocity_signal(poses, HAND_IDX, fps)

    if shoulder is None or forearm is None or core is None or hand is None:
        return None

    n = min(len(shoulder), len(forearm), len(core), len(hand))
    if n < 20:
        return None
    shoulder = shoulder[:n]
    forearm = forearm[:n]
    core = core[:n]
    hand = hand[:n]

    shoulder_peak_hz, shoulder_peak_power = _welch_peak_hz(shoulder, fs=fps)
    core_peak_hz, core_peak_power = _welch_peak_hz(core, fs=fps)
    plv, lag_frames, lag_seconds, lag_corr = _phase_metrics(shoulder, forearm, fps=fps)
    osc = _oscillator_fit(shoulder, forearm, dt=1.0 / max(fps, 1.0))
    delta_r2 = _predictor_delta_r2(shoulder, forearm, core, hand, max_lag=4)

    return {
        "shoulder_peak_hz": shoulder_peak_hz,
        "shoulder_peak_power": shoulder_peak_power,
        "core_peak_hz": core_peak_hz,
        "core_peak_power": core_peak_power,
        "plv": float(plv),
        "lead_lag_frames": int(lag_frames),
        "lead_lag_seconds": float(lag_seconds),
        "lead_lag_corr": float(lag_corr),
        "oscillator_r2_mean": float(0.5 * (osc["r2_upper"] + osc["r2_forearm"])),
        "oscillator_coupling_asymmetry": float(osc["coupling_asymmetry"]),
        "predictor_delta_r2": float(delta_r2),
    }


def _summarize_array(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
    }


def _hierarchical_bootstrap_ci(records, metric_key, n_boot, seed, alpha=0.05):
    vals_by_speaker = {}
    for rec in records:
        v = rec["metrics"].get(metric_key, np.nan)
        if not np.isfinite(v):
            continue
        sid = rec["speaker_id"]
        vals_by_speaker.setdefault(sid, []).append(float(v))

    speakers = list(vals_by_speaker.keys())
    if len(speakers) == 0:
        return {
            "n_sequences": 0,
            "n_speakers": 0,
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(int(n_boot)):
        sampled_speakers = [speakers[rng.randint(0, len(speakers))] for _ in range(len(speakers))]
        merged = []
        for sid in sampled_speakers:
            seq_vals = vals_by_speaker[sid]
            for _ in range(len(seq_vals)):
                merged.append(seq_vals[rng.randint(0, len(seq_vals))])
        if len(merged) > 0:
            boot_means.append(float(np.mean(merged)))

    if len(boot_means) == 0:
        return {
            "n_sequences": 0,
            "n_speakers": 0,
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    ql = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    qh = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "n_sequences": int(sum([len(v) for v in vals_by_speaker.values()])),
        "n_speakers": int(len(vals_by_speaker)),
        "mean": float(np.mean(boot_means)),
        "ci_low": ql,
        "ci_high": qh,
    }


def _sensitivity_report(combination_metrics):
    keys = [
        "shoulder_peak_hz",
        "core_peak_hz",
        "plv",
        "lead_lag_seconds",
        "oscillator_r2_mean",
        "predictor_delta_r2",
    ]
    out = {}
    for key in keys:
        vals = []
        for _, met in combination_metrics.items():
            v = met.get(key, np.nan)
            if np.isfinite(v):
                vals.append(float(v))
        if len(vals) == 0:
            out[key] = {"spread": np.nan, "cv": np.nan, "n": 0}
            continue
        arr = np.asarray(vals, dtype=np.float64)
        mean_abs = float(abs(np.mean(arr)))
        out[key] = {
            "spread": float(np.max(arr) - np.min(arr)),
            "cv": float(np.std(arr) / mean_abs) if mean_abs > 1e-9 else np.nan,
            "n": int(len(arr)),
        }
    return out


def _synthetic_sanity(seed=1234):
    rng = np.random.RandomState(seed)
    fps = 30.0
    duration = 20.0
    t = np.arange(int(duration * fps), dtype=np.float64) / fps

    target_f = 1.8
    phase = 0.45
    shoulder = np.sin(2.0 * np.pi * target_f * t)
    forearm = 0.8 * np.sin(2.0 * np.pi * target_f * t + phase) + 0.15 * np.sin(2.0 * np.pi * 2.0 * target_f * t)
    core = 0.45 * np.sin(2.0 * np.pi * 1.1 * t + 0.2)
    hand = 0.35 * shoulder + 0.55 * forearm + 0.25 * core + 0.05 * rng.randn(len(t))

    # Build pseudo-pose [T,55,3] so the same sequence pipeline is used.
    poses = np.zeros((len(t), 55, 3), dtype=np.float64)
    for j in [16, 17]:
        poses[:, j, 0] = shoulder
    for j in [18, 19, 20, 21]:
        poses[:, j, 0] = forearm
    for j in [20, 21]:
        poses[:, j, 1] = hand
    for j in [0, 3, 6, 9, 12]:
        poses[:, j, 2] = core

    met = _sequence_metrics(poses, fps=fps, shoulder_idx=[16, 17], core_idx=[0, 3, 6, 9, 12])
    if met is None:
        return {"status": "failed", "reason": "synthetic metrics returned None"}

    expected_lag_sec = phase / (2.0 * np.pi * target_f)
    lag_err = abs(abs(met["lead_lag_seconds"]) - expected_lag_sec)
    peak_err = abs(met["shoulder_peak_hz"] - target_f)

    return {
        "status": "ok",
        "target_frequency_hz": float(target_f),
        "expected_lag_seconds_abs": float(expected_lag_sec),
        "observed": met,
        "checks": {
            "peak_error_hz": float(peak_err),
            "lag_error_seconds": float(lag_err),
            "predictor_delta_positive": bool(met["predictor_delta_r2"] > 0.0),
            "plv_reasonable": bool(met["plv"] > 0.5),
            "peak_recovery_pass": bool(peak_err < 0.20),
            "lag_recovery_pass": bool(lag_err < 0.12),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Robust coupled-swing validation with bootstrap CIs and sensitivity checks.")
    parser.add_argument("--input_glob", type=str, required=True, help="Input npz glob (e.g., demo/*.npz)")
    parser.add_argument("--output", type=str, default="outputs/coupled_validation_report.json")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max_files", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if args.max_files > 0:
        files = files[: args.max_files]
    if len(files) == 0:
        raise ValueError("No files matched: {}".format(args.input_glob))

    all_seq_reports = {}
    default_records_by_cond = {}
    sensitivity_by_cond = {}

    shoulder_defs = DEFAULT_SHOULDER_DEFS
    core_defs = DEFAULT_CORE_DEFS
    default_shoulder_key = "shoulders_only"
    default_core_key = "torso_plus_neck"

    for path in files:
        poses, fps = _load_npz_pose(path)
        if poses is None:
            continue
        if poses.shape[0] < 24:
            continue

        seq_key = os.path.basename(path)
        speaker = _speaker_id_from_name(path)
        condition = _condition_from_name(path)

        combination_metrics = {}
        for sh_name, sh_idx in shoulder_defs.items():
            for core_name, core_idx in core_defs.items():
                combo_name = sh_name + "__" + core_name
                met = _sequence_metrics(poses, fps=fps, shoulder_idx=sh_idx, core_idx=core_idx)
                if met is not None:
                    combination_metrics[combo_name] = met

        if len(combination_metrics) == 0:
            continue

        default_name = default_shoulder_key + "__" + default_core_key
        if default_name not in combination_metrics:
            default_name = sorted(combination_metrics.keys())[0]
        default_metrics = combination_metrics[default_name]
        sensitivity = _sensitivity_report(combination_metrics)

        all_seq_reports[seq_key] = {
            "path": path,
            "speaker_id": speaker,
            "condition": condition,
            "fps": float(fps),
            "default_definition": default_name,
            "default_metrics": default_metrics,
            "sensitivity": sensitivity,
            "all_definitions": combination_metrics,
        }

        default_records_by_cond.setdefault(condition, []).append(
            {
                "speaker_id": speaker,
                "metrics": default_metrics,
            }
        )
        sensitivity_by_cond.setdefault(condition, []).append(sensitivity)

    metric_keys = [
        "shoulder_peak_hz",
        "core_peak_hz",
        "plv",
        "lead_lag_seconds",
        "oscillator_r2_mean",
        "predictor_delta_r2",
    ]

    bootstrap_report = {}
    triangulation = {}
    sensitivity_summary = {}
    for cond, records in default_records_by_cond.items():
        bootstrap_report[cond] = {}
        for mk in metric_keys:
            bootstrap_report[cond][mk] = _hierarchical_bootstrap_ci(
                records,
                metric_key=mk,
                n_boot=args.bootstrap,
                seed=args.seed + abs(hash((cond, mk))) % 100000,
                alpha=args.alpha,
            )

        vals_plv = [r["metrics"].get("plv", np.nan) for r in records]
        vals_delta = [r["metrics"].get("predictor_delta_r2", np.nan) for r in records]
        vals_r2 = [r["metrics"].get("oscillator_r2_mean", np.nan) for r in records]
        vals_peak = [r["metrics"].get("shoulder_peak_hz", np.nan) for r in records]

        vals_plv = np.asarray(vals_plv, dtype=np.float64)
        vals_delta = np.asarray(vals_delta, dtype=np.float64)
        vals_r2 = np.asarray(vals_r2, dtype=np.float64)
        vals_peak = np.asarray(vals_peak, dtype=np.float64)

        triangulation[cond] = {
            "n_sequences": int(len(records)),
            "shoulder_peak_hz": _summarize_array(vals_peak),
            "phase_locking_plv": _summarize_array(vals_plv),
            "oscillator_r2_mean": _summarize_array(vals_r2),
            "predictor_delta_r2": _summarize_array(vals_delta),
            "consistency": {
                "plv_gt_0_5_fraction": float(np.mean(vals_plv[np.isfinite(vals_plv)] > 0.5)) if np.sum(np.isfinite(vals_plv)) > 0 else np.nan,
                "predictor_delta_positive_fraction": float(np.mean(vals_delta[np.isfinite(vals_delta)] > 0.0)) if np.sum(np.isfinite(vals_delta)) > 0 else np.nan,
                "oscillator_r2_gt_0_3_fraction": float(np.mean(vals_r2[np.isfinite(vals_r2)] > 0.3)) if np.sum(np.isfinite(vals_r2)) > 0 else np.nan,
            },
        }

        spread_keys = [
            "shoulder_peak_hz",
            "core_peak_hz",
            "plv",
            "lead_lag_seconds",
            "oscillator_r2_mean",
            "predictor_delta_r2",
        ]
        cond_sens = sensitivity_by_cond.get(cond, [])
        sens_agg = {}
        for sk in spread_keys:
            spreads = [s[sk]["spread"] for s in cond_sens if sk in s and np.isfinite(s[sk]["spread"])]
            cvs = [s[sk]["cv"] for s in cond_sens if sk in s and np.isfinite(s[sk]["cv"])]
            sens_agg[sk] = {
                "spread": _summarize_array(spreads),
                "cv": _summarize_array(cvs),
            }
        sensitivity_summary[cond] = sens_agg

    synthetic = _synthetic_sanity(seed=args.seed + 101)

    report = {
        "summary": {
            "num_input_files": int(len(files)),
            "num_valid_sequences": int(len(all_seq_reports)),
            "conditions": sorted(list(default_records_by_cond.keys())),
            "default_joint_definitions": {
                "shoulder": default_shoulder_key,
                "core": default_core_key,
            },
            "all_shoulder_definitions": shoulder_defs,
            "all_core_definitions": core_defs,
        },
        "bootstrap_psd_and_estimators": bootstrap_report,
        "sensitivity_summary": sensitivity_summary,
        "synthetic_sanity": synthetic,
        "multi_estimator_triangulation": triangulation,
        "per_sequence": all_seq_reports,
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print("Saved validation report to {}".format(args.output))
    print("Input files: {} | Valid sequences: {}".format(len(files), len(all_seq_reports)))
    print("Conditions: {}".format(", ".join(sorted(list(default_records_by_cond.keys())))))


if __name__ == "__main__":
    main()