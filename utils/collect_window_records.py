"""
collect_window_records.py — gather per-window metrics from BEAT2 for plotting.

Produces a JSON file with a list of records, one per motion window, each with:
  category       : "beat" | "semantic"
  duration_s     : window duration in seconds
  shoulder_hz    : Welch PSD peak (shoulder joint angular velocity)
  arm_hz         : Welch PSD peak (arm combined angular velocity)
  hand_hz        : Welch PSD peak (hand joint angular velocity)
  plv            : Phase-Locking Value between shoulder and forearm
  osc_r2         : Coupled oscillator R² fit
  cycles         : shoulder_hz × duration_s (estimated # complete cycles in window)
  speaker_id     : speaker string (for bootstrap / colouring)
"""

import argparse
import json
import os

import numpy as np

# Re-use helper functions from the existing validation script.
try:
    from utils import run_gt_semantic_beat_validation as gt
except ModuleNotFoundError:
    import run_gt_semantic_beat_validation as gt


HAND_IDX = [20, 21]


def _collect_per_window(npz_path, sem_path, min_window_frames, speaker_id, clip_id):
    poses, fps = gt._load_motion(npz_path)
    if poses is None or poses.shape[0] < min_window_frames + 5:
        return []

    labels, _ = gt._load_semantic_labels(sem_path, poses.shape[0], fps)

    # Pre-compute signals once per clip.
    sig_shoulder = gt._angular_velocity_signal(poses, gt.SHOULDER_IDX, fps)
    sig_forearm = gt._angular_velocity_signal(poses, gt.FOREARM_IDX, fps)
    sig_arm = gt._angular_velocity_signal(poses, gt.ARM_COMBINED_IDX, fps)
    sig_hand = gt._angular_velocity_signal(poses, HAND_IDX, fps)
    if any(s is None for s in [sig_shoulder, sig_forearm, sig_arm, sig_hand]):
        return []

    records = []
    for cat_int, cat_name in [(1, "beat"), (2, "semantic")]:
        windows = gt._find_windows(labels, cat_int, min_window_frames)
        for start, end in windows:
            n = end - start
            s_sh = sig_shoulder[start:end]
            s_fa = sig_forearm[start:end]
            s_arm = sig_arm[start:end]
            s_hand = sig_hand[start:end]

            if min(len(s_sh), len(s_fa)) < 12:
                continue

            sh_hz, _ = gt._welch_peak_hz(s_sh, fs=fps)
            arm_hz, _ = gt._welch_peak_hz(s_arm, fs=fps)
            hand_hz, _ = gt._welch_peak_hz(s_hand, fs=fps)
            plv, _, _, _ = gt._phase_metrics(s_sh, s_fa, fps=fps)
            osc_r2 = gt._oscillator_fit(s_sh, s_fa, dt=1.0 / max(fps, 1.0))

            dur_s = float(n) / float(fps)
            cycles = float(sh_hz) * dur_s if np.isfinite(sh_hz) else float("nan")

            records.append({
                "category": cat_name,
                "speaker_id": str(speaker_id),
                "clip_id": str(clip_id),
                "duration_s": round(dur_s, 4),
                "shoulder_hz": round(float(sh_hz), 4) if np.isfinite(sh_hz) else None,
                "arm_hz": round(float(arm_hz), 4) if np.isfinite(arm_hz) else None,
                "hand_hz": round(float(hand_hz), 4) if np.isfinite(hand_hz) else None,
                "plv": round(float(plv), 4) if np.isfinite(plv) else None,
                "osc_r2": round(float(osc_r2), 4) if np.isfinite(osc_r2) else None,
                "cycles": round(cycles, 3) if np.isfinite(cycles) else None,
            })
    return records


def main():
    parser = argparse.ArgumentParser(description="Collect per-window metrics for beat/semantic plots.")
    parser.add_argument("--beat2_dir", type=str, default="BEAT2/beat_english_v2.0.0")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--min_window_frames", type=int, default=15)
    parser.add_argument("--output", type=str, default="outputs/window_records.json")
    args = parser.parse_args()

    motion_dir = os.path.join(args.beat2_dir, "smplxflame_30")
    sem_dir = os.path.join(args.beat2_dir, "sem")
    split_csv = os.path.join(args.beat2_dir, "train_test_split.csv")

    allowed = gt._load_allowed_split_ids(split_csv, args.split)

    all_records = []
    npz_files = sorted([f for f in os.listdir(motion_dir) if f.endswith(".npz")])
    processed = 0

    for fname in npz_files:
        clip_id = fname[:-4]
        if allowed is not None and clip_id not in allowed:
            continue
        sem_path = os.path.join(sem_dir, clip_id + ".txt")
        if not os.path.exists(sem_path):
            continue
        speaker_id = clip_id.split("_")[0]
        records = _collect_per_window(
            os.path.join(motion_dir, fname), sem_path,
            args.min_window_frames, speaker_id, clip_id
        )
        all_records.extend(records)
        processed += 1
        if processed % 50 == 0:
            print(f"  processed {processed} clips, {len(all_records)} windows so far...")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_records, f)

    beat_n = sum(1 for r in all_records if r["category"] == "beat")
    sem_n = sum(1 for r in all_records if r["category"] == "semantic")
    print(f"Done. {processed} clips → {len(all_records)} windows ({beat_n} beat, {sem_n} semantic)")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
