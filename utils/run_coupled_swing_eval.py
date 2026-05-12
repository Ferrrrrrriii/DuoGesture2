import argparse
import glob
import json
import os

import numpy as np

from utils.metric import CoupledSwingAnalyzer


def _speaker_id_from_name(path):
    name = os.path.basename(path)
    token = name.split("_")[0]
    try:
        return int(token)
    except Exception:
        return -1


def _load_npz_key(npz_obj, key):
    if key is None:
        return None
    if key in npz_obj:
        return np.asarray(npz_obj[key])
    return None


def _combine_signals(sig_r, sig_l):
    out = {}
    for key in ["shoulder", "forearm", "torso", "hand"]:
        xr = np.asarray(sig_r[key]).reshape(-1)
        xl = np.asarray(sig_l[key]).reshape(-1)
        n = min(len(xr), len(xl))
        out[key] = 0.5 * (xr[:n] + xl[:n])
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate coupled swing dynamics from joint trajectories.")
    parser.add_argument("--input_glob", type=str, required=True, help="Glob for input npz files.")
    parser.add_argument("--joints_key", type=str, default="joints", help="Key in npz containing [T,J,3] joints.")
    parser.add_argument("--audio_key", type=str, default="audio_onset", help="Optional key for audio onset/beat signal.")
    parser.add_argument("--gate_key", type=str, default="semantic_gate", help="Optional key for semantic gate signal.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max_lag_pred", type=int, default=4)
    parser.add_argument("--max_lag_audio", type=int, default=8)
    parser.add_argument("--output", type=str, default="outputs/coupled_swing_report.json")
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if len(files) == 0:
        raise ValueError("No files matched --input_glob: {}".format(args.input_glob))

    analyzer = CoupledSwingAnalyzer(fps=args.fps)
    per_sequence = {}
    predictor_samples = []

    for path in files:
        with np.load(path, allow_pickle=True) as arr:
            joints = _load_npz_key(arr, args.joints_key)
            if joints is None:
                continue
            if joints.ndim != 3 or joints.shape[-1] != 3:
                continue

            left = analyzer.side_report(joints, side="left")
            right = analyzer.side_report(joints, side="right")
            asym = analyzer.bilateral_oscillator_asymmetry(joints)

            key = os.path.basename(path)
            per_sequence[key] = {
                "left": left,
                "right": right,
                "bilateral_asymmetry": asym,
            }

            sig_r = analyzer.extract_predictor_signals(joints, side="right")
            sig_l = analyzer.extract_predictor_signals(joints, side="left")
            combined = _combine_signals(sig_r, sig_l)

            audio = _load_npz_key(arr, args.audio_key)
            gate = _load_npz_key(arr, args.gate_key)
            if audio is not None:
                audio = np.asarray(audio).reshape(-1)
            if gate is not None:
                gate = np.asarray(gate).reshape(-1)

            sample = {
                "speaker_id": _speaker_id_from_name(path),
                "shoulder": combined["shoulder"],
                "forearm": combined["forearm"],
                "torso": combined["torso"],
                "hand": combined["hand"],
                "audio": audio,
                "gate": gate,
            }
            predictor_samples.append(sample)

    predictor_report = analyzer.predictor_analysis(predictor_samples, max_lag=args.max_lag_pred)
    causality_report = analyzer.audio_conditioned_causality(predictor_samples, max_lag=args.max_lag_audio)

    summary = {
        "num_input_files": len(files),
        "num_valid_sequences": len(per_sequence),
        "predictor_analysis": predictor_report,
        "audio_conditioned_causality": causality_report,
    }

    out = {
        "summary": summary,
        "per_sequence": per_sequence,
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print("Saved coupled swing report to {}".format(args.output))
    print("Valid sequences: {}".format(len(per_sequence)))


if __name__ == "__main__":
    main()
