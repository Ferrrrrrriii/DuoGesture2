import argparse
import glob
import json
import os

try:
    from utils import run_gt_semantic_beat_validation as gt
except ModuleNotFoundError:
    # Support direct script execution: python utils/run_generated_semantic_beat_eval.py
    import run_gt_semantic_beat_validation as gt


def _clip_id_from_generated(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.startswith("res_"):
        return stem[4:]
    if stem.startswith("gt_"):
        return stem[3:]
    if stem.startswith("gen_"):
        return stem[4:]
    return stem


def _keep_generated(path):
    name = os.path.basename(path)
    # Prefer generated predictions; skip explicit GT dumps if present.
    return not name.startswith("gt_")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated NPZ motions with GT semantic/beat windows from BEAT2 labels."
    )
    parser.add_argument("--generated_glob", type=str, required=True)
    parser.add_argument("--beat2_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--min_window_frames", type=int, default=15)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--output", type=str, default="outputs/generated_semantic_beat_eval.json")
    args = parser.parse_args()

    files = sorted([p for p in glob.glob(args.generated_glob) if _keep_generated(p)])
    if args.max_files > 0:
        files = files[: args.max_files]
    if len(files) == 0:
        raise ValueError("No generated files matched --generated_glob")

    sem_dir = os.path.join(args.beat2_dir, "sem")
    split_csv = os.path.join(args.beat2_dir, "train_test_split.csv")
    allowed = gt._load_allowed_split_ids(split_csv, args.split)

    all_window_records = []
    per_file_summary = {}

    for path in files:
        clip_id = _clip_id_from_generated(path)
        if allowed is not None and clip_id not in allowed:
            continue

        sem_path = os.path.join(sem_dir, clip_id + ".txt")
        if not os.path.exists(sem_path):
            continue

        poses, fps = gt._load_motion(path)
        if poses is None or poses.shape[0] < args.min_window_frames + 5:
            continue

        labels, _ = gt._load_semantic_labels(sem_path, poses.shape[0], fps)

        sig_shoulder = gt._angular_velocity_signal(poses, gt.SHOULDER_IDX, fps)
        sig_forearm = gt._angular_velocity_signal(poses, gt.FOREARM_IDX, fps)
        sig_arm = gt._angular_velocity_signal(poses, gt.ARM_COMBINED_IDX, fps)
        sig_core = gt._angular_velocity_signal(poses, gt.CORE_IDX, fps)
        sig_hand = gt._angular_velocity_signal(poses, gt.HAND_IDX, fps)
        sig_all = gt._angular_velocity_signal(poses, list(range(min(55, poses.shape[1]))), fps)
        if any([s is None for s in [sig_shoulder, sig_forearm, sig_arm, sig_core, sig_hand, sig_all]]):
            continue

        n = min(
            len(sig_shoulder),
            len(sig_forearm),
            len(sig_arm),
            len(sig_core),
            len(sig_hand),
            len(sig_all),
            len(labels) - 1,
        )
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

        beat_windows = gt._find_windows(labels_sig, 1, args.min_window_frames)
        sem_windows = gt._find_windows(labels_sig, 2, args.min_window_frames)

        c_records = []
        speaker_id = clip_id.split("_")[0]
        for start, end in beat_windows:
            rec = gt._window_metric_record(signals, start, end, fps, speaker_id, clip_id, "beat")
            if rec is not None:
                all_window_records.append(rec)
                c_records.append(rec)
        for start, end in sem_windows:
            rec = gt._window_metric_record(signals, start, end, fps, speaker_id, clip_id, "semantic")
            if rec is not None:
                all_window_records.append(rec)
                c_records.append(rec)

        per_file_summary[os.path.basename(path)] = {
            "clip_id": clip_id,
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
            "duration_seconds": gt._summarize([r["duration_seconds"] for r in records]),
            "metrics": {},
            "duration_bins_peak_hz": gt._duration_bins(records, "shoulder_peak_hz"),
        }
        for mk in metric_keys:
            vals = [r.get(mk, float("nan")) for r in records]
            wts = [r.get("duration_seconds", float("nan")) for r in records]
            cat_reports[cat]["metrics"][mk] = {
                "window_mean": gt._summarize(vals),
                "duration_weighted": {
                    "mean": gt._weighted_mean(vals, wts),
                },
                "bootstrap_window_mean": gt._hierarchical_bootstrap(
                    records,
                    mk,
                    args.bootstrap,
                    args.seed + abs(hash((cat, mk, "uw"))) % 100000,
                    weighted=False,
                ),
                "bootstrap_duration_weighted": gt._hierarchical_bootstrap(
                    records,
                    mk,
                    args.bootstrap,
                    args.seed + abs(hash((cat, mk, "w"))) % 100000,
                    weighted=True,
                ),
            }

    beat_feat = gt._feature_group_recommendation(cat_reports.get("beat", {}).get("metrics", {}))
    sem_feat = gt._feature_group_recommendation(cat_reports.get("semantic", {}).get("metrics", {}))
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
            "generated_glob": args.generated_glob,
            "beat2_dir": args.beat2_dir,
            "split": args.split,
            "n_input_files": int(len(files)),
            "n_processed_files": int(len(per_file_summary)),
            "n_total_metric_windows": int(len(all_window_records)),
            "n_beat_windows": int(len(by_cat["beat"])),
            "n_semantic_windows": int(len(by_cat["semantic"])),
            "min_window_frames": int(args.min_window_frames),
        },
        "category_reports": cat_reports,
        "feature_group_recommendation": overall_rec,
        "per_file_summary": per_file_summary,
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print("Saved generated-vs-GT semantic/beat report to {}".format(args.output))
    print("Processed files: {} | total windows: {}".format(len(per_file_summary), len(all_window_records)))


if __name__ == "__main__":
    main()
