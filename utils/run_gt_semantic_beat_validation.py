import argparse
import json
import os
import random

import numpy as np
from scipy.signal import hilbert, welch


FPS_DEFAULT = 30.0

SHOULDER_IDX = [16, 17]
FOREARM_IDX = [18, 19, 20, 21]
ARM_COMBINED_IDX = [16, 17, 18, 19, 20, 21]
CORE_IDX = [0, 3, 6, 9, 12]
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
        return np.nan
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return np.nan
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


def _load_motion(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    if "poses" in d:
        poses = np.asarray(d["poses"])
    elif "pose" in d:
        poses = np.asarray(d["pose"])
    else:
        return None, None
    fps = float(np.asarray(d["mocap_frame_rate"]).reshape(-1)[0]) if "mocap_frame_rate" in d else FPS_DEFAULT
    if poses.ndim == 2 and poses.shape[-1] % 3 == 0:
        poses = poses.reshape(poses.shape[0], poses.shape[-1] // 3, 3)
    if poses.ndim != 3 or poses.shape[-1] != 3:
        return None, None
    return poses.astype(np.float64), fps


def _load_semantic_labels(sem_path, n_frames, fps):
    labels = np.zeros(n_frames, dtype=np.int32)
    kinds = [""] * n_frames
    if not os.path.exists(sem_path):
        return labels, kinds

    with open(sem_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                parts = line.split()
            if len(parts) < 5:
                continue
            name = str(parts[0]).strip().lower()
            try:
                start_t = float(parts[1])
                end_t = float(parts[2])
                score = float(parts[4])
            except Exception:
                continue

            sf = max(0, int(np.floor(start_t * fps)))
            ef = min(n_frames, int(np.ceil(end_t * fps)))
            if ef <= sf:
                continue

            if score <= 0.1 or "beat" in name:
                cat = 1
            else:
                cat = 2

            labels[sf:ef] = cat
            for i in range(sf, ef):
                kinds[i] = name

    return labels, kinds


def _find_windows(labels, target, min_len):
    windows = []
    in_seg = False
    start = 0
    for i in range(len(labels)):
        if labels[i] == target and not in_seg:
            start = i
            in_seg = True
        elif labels[i] != target and in_seg:
            if i - start >= min_len:
                windows.append((start, i))
            in_seg = False
    if in_seg and len(labels) - start >= min_len:
        windows.append((start, len(labels)))
    return windows


def _angular_velocity_signal(poses, joint_indices, fps):
    idx = [i for i in joint_indices if i < poses.shape[1]]
    if len(idx) == 0 or poses.shape[0] < 4:
        return None
    omega = np.diff(poses[:, idx, :], axis=0) * float(fps)
    flat = omega.reshape(omega.shape[0], -1)
    flat = flat - np.mean(flat, axis=0, keepdims=True)
    if flat.shape[1] == 1:
        return flat[:, 0]
    try:
        _, _, vh = np.linalg.svd(flat, full_matrices=False)
        sig = flat @ vh[0]
        if np.std(sig) < 1e-10:
            return np.mean(flat, axis=1)
        return sig
    except Exception:
        return np.mean(flat, axis=1)


def _welch_peak_hz(sig, fs, f_min=0.3, f_max=8.0):
    sig = np.asarray(sig).reshape(-1)
    if len(sig) < 12:
        return np.nan, np.nan
    sig = sig - np.mean(sig)
    nperseg = int(min(128, len(sig)))
    noverlap = int(max(0, nperseg // 2))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="linear", window="hann")
    band = (freqs >= f_min) & (freqs <= f_max)
    if np.sum(band) < 2:
        return np.nan, np.nan
    fb = freqs[band]
    pb = psd[band]
    idx = int(np.argmax(pb))
    return float(fb[idx]), float(pb[idx])


def _phase_metrics(x, y, fps, max_lag_s=0.6):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    n = min(len(x), len(y))
    if n < 8:
        return np.nan, np.nan, np.nan, np.nan
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
        return np.nan
    x1 = x1[:n]
    x2 = x2[:n]
    dx1 = np.gradient(x1, dt)
    dx2 = np.gradient(x2, dt)
    ddx1 = np.gradient(dx1, dt)
    ddx2 = np.gradient(dx2, dt)

    X1 = np.stack([dx1, x1, (x1 - x2)], axis=1)
    y1 = -ddx1
    _, y1_hat = _fit_linear(X1, y1)

    X2 = np.stack([dx2, x2, (x2 - x1)], axis=1)
    y2 = -ddx2
    _, y2_hat = _fit_linear(X2, y2)

    r1 = _r2(y1, y1_hat)
    r2 = _r2(y2, y2_hat)
    vals = [v for v in [r1, r2] if np.isfinite(v)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def _make_lag_features(signals, keys, max_lag):
    arrays = [np.asarray(signals[k]).reshape(-1) for k in keys if k in signals and signals[k] is not None]
    if len(arrays) == 0:
        return None, None
    n = min([len(a) for a in arrays])
    if n <= max_lag + 2:
        return None, None
    feats = []
    for k in keys:
        if k not in signals or signals[k] is None:
            continue
        s = np.asarray(signals[k]).reshape(-1)[:n]
        for lag in range(max_lag + 1):
            feats.append(s[max_lag - lag:n - lag])
    X = np.stack(feats, axis=1)
    y = np.asarray(signals["hand"]).reshape(-1)[:n][max_lag:]
    return X, y


def _predictor_r2(signals, keys, max_lag=4):
    X, y = _make_lag_features(signals, keys, max_lag)
    if X is None:
        return np.nan

    n = X.shape[0]
    split = int(0.7 * n)
    if split <= 8 or (n - split) <= 8:
        return np.nan

    Xtr, ytr = X[:split], y[:split]
    Xte, yte = X[split:], y[split:]
    w, _ = _fit_linear(Xtr, ytr)
    Xteb = np.concatenate([Xte, np.ones((Xte.shape[0], 1), dtype=np.float64)], axis=1)
    yhat = Xteb @ w
    return _r2(yte, yhat)


def _window_metric_record(signals, start, end, fps, speaker_id, clip_id, category):
    sub = {}
    for k, v in signals.items():
        if v is None:
            return None
        if end > len(v):
            return None
        sub[k] = v[start:end]

    n = min([len(v) for v in sub.values()])
    if n < 20:
        return None
    for k in sub.keys():
        sub[k] = sub[k][:n]

    shoulder_peak_hz, shoulder_peak_power = _welch_peak_hz(sub["shoulder"], fs=fps)
    arm_peak_hz, arm_peak_power = _welch_peak_hz(sub["arm_combined"], fs=fps)
    plv, lag_frames, lag_seconds, lag_corr = _phase_metrics(sub["shoulder"], sub["forearm"], fps=fps)
    osc_r2 = _oscillator_fit(sub["shoulder"], sub["forearm"], dt=1.0 / max(fps, 1.0))

    r2_shoulder = _predictor_r2(sub, ["shoulder"], max_lag=4)
    r2_split = _predictor_r2(sub, ["shoulder", "forearm", "core"], max_lag=4)
    r2_arm_combined = _predictor_r2(sub, ["arm_combined", "core"], max_lag=4)
    r2_all = _predictor_r2(sub, ["all_joints", "core"], max_lag=4)

    return {
        "speaker_id": speaker_id,
        "clip_id": clip_id,
        "category": category,
        "duration_frames": int(n),
        "duration_seconds": float(n / float(fps)),
        "shoulder_peak_hz": shoulder_peak_hz,
        "shoulder_peak_power": shoulder_peak_power,
        "arm_peak_hz": arm_peak_hz,
        "arm_peak_power": arm_peak_power,
        "plv": plv,
        "lead_lag_frames": lag_frames,
        "lead_lag_seconds": lag_seconds,
        "lead_lag_corr": lag_corr,
        "oscillator_r2_mean": osc_r2,
        "r2_shoulder_only": r2_shoulder,
        "r2_split_arm_core": r2_split,
        "r2_arm_combined_core": r2_arm_combined,
        "r2_all_joints_core": r2_all,
        "delta_r2_split_vs_shoulder": float(r2_split - r2_shoulder) if np.isfinite(r2_split) and np.isfinite(r2_shoulder) else np.nan,
        "delta_r2_arm_combined_vs_shoulder": float(r2_arm_combined - r2_shoulder) if np.isfinite(r2_arm_combined) and np.isfinite(r2_shoulder) else np.nan,
        "delta_r2_all_joints_vs_shoulder": float(r2_all - r2_shoulder) if np.isfinite(r2_all) and np.isfinite(r2_shoulder) else np.nan,
    }


def _summarize(values):
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


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if np.sum(mask) == 0:
        return np.nan
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def _hierarchical_bootstrap(records, key, n_boot, seed, weighted=False, alpha=0.05):
    nested = {}
    for r in records:
        v = r.get(key, np.nan)
        if not np.isfinite(v):
            continue
        sid = r["speaker_id"]
        cid = r["clip_id"]
        nested.setdefault(sid, {})
        nested[sid].setdefault(cid, [])
        nested[sid][cid].append((float(v), float(r["duration_seconds"])))

    speakers = list(nested.keys())
    if len(speakers) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_speakers": 0, "n_windows": 0}

    rng = np.random.RandomState(seed)
    boot_vals = []
    for _ in range(int(n_boot)):
        sampled_speakers = [speakers[rng.randint(0, len(speakers))] for _ in range(len(speakers))]
        vals = []
        wts = []
        for sid in sampled_speakers:
            clips = list(nested[sid].keys())
            sampled_clips = [clips[rng.randint(0, len(clips))] for _ in range(len(clips))]
            for cid in sampled_clips:
                wins = nested[sid][cid]
                sampled_wins = [wins[rng.randint(0, len(wins))] for _ in range(len(wins))]
                for v, w in sampled_wins:
                    vals.append(v)
                    wts.append(w)
        if len(vals) == 0:
            continue
        if weighted:
            m = _weighted_mean(vals, wts)
        else:
            m = float(np.mean(vals))
        if np.isfinite(m):
            boot_vals.append(m)

    if len(boot_vals) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_speakers": len(speakers), "n_windows": 0}

    return {
        "mean": float(np.mean(boot_vals)),
        "ci_low": float(np.percentile(boot_vals, 100.0 * (alpha / 2.0))),
        "ci_high": float(np.percentile(boot_vals, 100.0 * (1.0 - alpha / 2.0))),
        "n_speakers": int(len(speakers)),
        "n_windows": int(sum([len(vv) for s in nested.values() for vv in s.values()])),
    }


def _duration_bins(records, key):
    bins = {
        "short_lt_1s": [],
        "mid_1_to_2p5s": [],
        "long_gt_2p5s": [],
    }
    for r in records:
        v = r.get(key, np.nan)
        d = r.get("duration_seconds", np.nan)
        if not np.isfinite(v) or not np.isfinite(d):
            continue
        if d < 1.0:
            bins["short_lt_1s"].append(v)
        elif d <= 2.5:
            bins["mid_1_to_2p5s"].append(v)
        else:
            bins["long_gt_2p5s"].append(v)
    return {k: _summarize(v) for k, v in bins.items()}


def _feature_group_recommendation(metrics):
    split_gain = metrics.get("delta_r2_split_vs_shoulder", {}).get("duration_weighted", {}).get("mean", np.nan)
    arm_gain = metrics.get("delta_r2_arm_combined_vs_shoulder", {}).get("duration_weighted", {}).get("mean", np.nan)
    all_gain = metrics.get("delta_r2_all_joints_vs_shoulder", {}).get("duration_weighted", {}).get("mean", np.nan)

    vals = {
        "split_arm_core": split_gain,
        "arm_combined_core": arm_gain,
        "all_joints_core": all_gain,
    }
    finite = {k: v for k, v in vals.items() if np.isfinite(v)}
    if len(finite) == 0:
        return {"best": "undetermined", "reason": "insufficient valid predictor windows"}

    best = sorted(finite.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    return {
        "best": best,
        "delta_r2": float(finite[best]),
        "all": finite,
    }


def _collect_valid_clip_ids(motion_dir, sem_dir):
    motion = [f[:-4] for f in os.listdir(motion_dir) if f.endswith(".npz")]
    sem = set([f[:-4] for f in os.listdir(sem_dir) if f.endswith(".txt")])
    valid = [c for c in motion if c in sem]
    return sorted(valid)


def _sample_per_speaker(clip_ids, n_per_speaker, seed):
    by_spk = {}
    for c in clip_ids:
        sid = c.split("_")[0]
        by_spk.setdefault(sid, []).append(c)

    rng = random.Random(seed)
    selected = []
    for sid, clips in sorted(by_spk.items()):
        if n_per_speaker <= 0:
            selected.extend(sorted(clips))
        else:
            k = min(n_per_speaker, len(clips))
            selected.extend(sorted(rng.sample(clips, k)))
    return selected


def _load_allowed_split_ids(split_csv_path, split_name):
    if split_name == "all":
        return None
    if not os.path.exists(split_csv_path):
        return None

    allowed = set()
    with open(split_csv_path, "r") as f:
        header = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header:
                header = False
                if line.lower().startswith("id,"):
                    continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            cid = parts[0].strip()
            ctype = parts[1].strip().lower()
            if ctype == split_name:
                allowed.add(cid)
    return allowed


def _apply_split_filter(clip_ids, allowed):
    if allowed is None:
        return clip_ids
    return [c for c in clip_ids if c in allowed]


def main():
    parser = argparse.ArgumentParser(description="Ground-truth beat-vs-semantic validation on BEAT2 with variable window-aware analysis.")
    parser.add_argument("--beat2_dir", type=str, required=True, help="Path to BEAT2 language folder, e.g. BEAT2/beat_english_v2.0.0")
    parser.add_argument("--n_per_speaker", type=int, default=4, help="Clips per speaker; 0 means all")
    parser.add_argument("--min_window_frames", type=int, default=15, help="Minimum contiguous window length in frames")
    parser.add_argument("--bootstrap", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"], help="Dataset split filter from train_test_split.csv")
    parser.add_argument("--max_windows_per_clip", type=int, default=0, help="0 means all")
    parser.add_argument("--output", type=str, default="outputs/gt_semantic_beat_validation.json")
    args = parser.parse_args()

    motion_dir = os.path.join(args.beat2_dir, "smplxflame_30")
    sem_dir = os.path.join(args.beat2_dir, "sem")
    if not os.path.isdir(motion_dir) or not os.path.isdir(sem_dir):
        raise ValueError("Invalid BEAT2 directory structure under {}".format(args.beat2_dir))

    split_csv = os.path.join(args.beat2_dir, "train_test_split.csv")
    clip_ids = _collect_valid_clip_ids(motion_dir, sem_dir)
    allowed = _load_allowed_split_ids(split_csv, args.split)
    clip_ids = _apply_split_filter(clip_ids, allowed)
    clip_ids = _sample_per_speaker(clip_ids, args.n_per_speaker, args.seed)
    if len(clip_ids) == 0:
        raise ValueError("No valid clips found with both motion and semantic files")

    all_window_records = []
    per_clip_summary = {}

    for cid in clip_ids:
        npz_path = os.path.join(motion_dir, cid + ".npz")
        sem_path = os.path.join(sem_dir, cid + ".txt")
        poses, fps = _load_motion(npz_path)
        if poses is None or poses.shape[0] < args.min_window_frames + 5:
            continue

        speaker_id = cid.split("_")[0]
        labels, _ = _load_semantic_labels(sem_path, poses.shape[0], fps)

        sig_shoulder = _angular_velocity_signal(poses, SHOULDER_IDX, fps)
        sig_forearm = _angular_velocity_signal(poses, FOREARM_IDX, fps)
        sig_arm = _angular_velocity_signal(poses, ARM_COMBINED_IDX, fps)
        sig_core = _angular_velocity_signal(poses, CORE_IDX, fps)
        sig_hand = _angular_velocity_signal(poses, HAND_IDX, fps)
        sig_all = _angular_velocity_signal(poses, list(range(min(55, poses.shape[1]))), fps)
        if any([s is None for s in [sig_shoulder, sig_forearm, sig_arm, sig_core, sig_hand, sig_all]]):
            continue

        n = min(len(sig_shoulder), len(sig_forearm), len(sig_arm), len(sig_core), len(sig_hand), len(sig_all), len(labels) - 1)
        if n < args.min_window_frames:
            continue

        labels_sig = labels[1:1 + n]
        signals = {
            "shoulder": sig_shoulder[:n],
            "forearm": sig_forearm[:n],
            "arm_combined": sig_arm[:n],
            "core": sig_core[:n],
            "hand": sig_hand[:n],
            "all_joints": sig_all[:n],
        }

        beat_windows = _find_windows(labels_sig, 1, args.min_window_frames)
        sem_windows = _find_windows(labels_sig, 2, args.min_window_frames)

        if args.max_windows_per_clip > 0:
            beat_windows = beat_windows[: args.max_windows_per_clip]
            sem_windows = sem_windows[: args.max_windows_per_clip]

        c_records = []
        for start, end in beat_windows:
            rec = _window_metric_record(signals, start, end, fps, speaker_id, cid, "beat")
            if rec is not None:
                all_window_records.append(rec)
                c_records.append(rec)
        for start, end in sem_windows:
            rec = _window_metric_record(signals, start, end, fps, speaker_id, cid, "semantic")
            if rec is not None:
                all_window_records.append(rec)
                c_records.append(rec)

        per_clip_summary[cid] = {
            "speaker_id": speaker_id,
            "fps": float(fps),
            "n_beat_windows": int(len(beat_windows)),
            "n_semantic_windows": int(len(sem_windows)),
            "n_valid_metric_windows": int(len(c_records)),
        }

    by_cat = {
        "beat": [r for r in all_window_records if r["category"] == "beat"],
        "semantic": [r for r in all_window_records if r["category"] == "semantic"],
    }

    metric_keys = [
        "shoulder_peak_hz",
        "arm_peak_hz",
        "plv",
        "lead_lag_seconds",
        "oscillator_r2_mean",
        "delta_r2_split_vs_shoulder",
        "delta_r2_arm_combined_vs_shoulder",
        "delta_r2_all_joints_vs_shoulder",
    ]

    cat_reports = {}
    for cat, records in by_cat.items():
        cat_reports[cat] = {
            "num_windows": int(len(records)),
            "duration_seconds": _summarize([r["duration_seconds"] for r in records]),
            "metrics": {},
            "duration_bins_peak_hz": _duration_bins(records, "shoulder_peak_hz"),
        }
        for mk in metric_keys:
            vals = [r.get(mk, np.nan) for r in records]
            wts = [r.get("duration_seconds", np.nan) for r in records]
            cat_reports[cat]["metrics"][mk] = {
                "window_mean": _summarize(vals),
                "duration_weighted": {
                    "mean": _weighted_mean(vals, wts),
                },
                "bootstrap_window_mean": _hierarchical_bootstrap(records, mk, args.bootstrap, args.seed + abs(hash((cat, mk, "uw"))) % 100000, weighted=False),
                "bootstrap_duration_weighted": _hierarchical_bootstrap(records, mk, args.bootstrap, args.seed + abs(hash((cat, mk, "w"))) % 100000, weighted=True),
            }

    beat_feat = _feature_group_recommendation(cat_reports.get("beat", {}).get("metrics", {}))
    sem_feat = _feature_group_recommendation(cat_reports.get("semantic", {}).get("metrics", {}))

    overall_rec = {
        "beat": beat_feat,
        "semantic": sem_feat,
    }
    if beat_feat.get("best") == sem_feat.get("best") and beat_feat.get("best") != "undetermined":
        overall_rec["combined"] = {
            "recommended_feature_group": beat_feat.get("best"),
            "rationale": "same best group for beat and semantic windows",
        }
    else:
        overall_rec["combined"] = {
            "recommended_feature_group": "hybrid",
            "rationale": "best feature group differs between beat and semantic windows",
        }

    report = {
        "summary": {
            "beat2_dir": args.beat2_dir,
            "n_selected_clips": int(len(clip_ids)),
            "split": args.split,
            "n_processed_clips": int(len(per_clip_summary)),
            "n_total_metric_windows": int(len(all_window_records)),
            "n_beat_windows": int(len(by_cat["beat"])),
            "n_semantic_windows": int(len(by_cat["semantic"])),
            "min_window_frames": int(args.min_window_frames),
            "notes": [
                "Window lengths differ strongly; both unweighted and duration-weighted statistics are reported.",
                "Bootstrap uses hierarchical speaker->clip->window resampling.",
                "Frequency metrics use Welch PSD per window with adaptive nperseg.",
            ],
        },
        "category_reports": cat_reports,
        "feature_group_recommendation": overall_rec,
        "per_clip_summary": per_clip_summary,
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print("Saved GT beat/semantic validation report to {}".format(args.output))
    print("Processed clips: {} | total windows: {}".format(len(per_clip_summary), len(all_window_records)))


if __name__ == "__main__":
    main()