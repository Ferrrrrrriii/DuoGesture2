#!/usr/bin/env python3
"""
Run fair, DuoGesture-style evaluation for a list of checkpoints.

Protocol parity with repo DuoGesture testing:
- Uses train.py --test_state (eval-only)
- Uses the provided config and training_speakers filter
- Parses metrics directly from trainer logs:
  l2 loss, lvel loss, fid score, align score, l1div score, total inference time

Usage:
    python utils/run_fair_duogesture_eval.py \
    --config configs/duogesture_base.yaml \
    --candidates outputs/findings/fair_eval_candidates_2026-04-05.json \
    --training_speakers 2 \
    --top_k 3
"""

import argparse
import csv
import datetime
import json
import os
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


def _load_candidates(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Candidates file must contain a JSON list")

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError("Each candidate must be an object; bad index: %d" % idx)
        name = item.get("name")
        ckpt = item.get("ckpt")
        extra_args = item.get("extra_args", [])
        if not name or not ckpt:
            raise ValueError("Each candidate needs 'name' and 'ckpt'; bad index: %d" % idx)
        if not isinstance(extra_args, list):
            raise ValueError("'extra_args' must be a list; bad candidate: %s" % name)
        normalized.append(
            {
                "name": str(name),
                "ckpt": str(ckpt),
                "extra_args": [str(x) for x in extra_args],
                "notes": str(item.get("notes", "_fair_eval_%s" % str(name))),
            }
        )
    return normalized


def _last_float(pattern: str, text: str) -> Optional[float]:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    val = matches[-1]
    if isinstance(val, tuple):
        val = val[0]
    try:
        return float(val)
    except ValueError:
        return None


def _last_inference_time(text: str) -> Tuple[Optional[int], Optional[int]]:
    # Example: total inference time: 89 s for 956 s motion
    matches = re.findall(
        r"total inference time:\s*([0-9]+)\s*s\s*for\s*([0-9]+)\s*s motion",
        text,
        flags=re.MULTILINE,
    )
    if not matches:
        return None, None
    infer_s, motion_s = matches[-1]
    return int(infer_s), int(motion_s)


def _extract_error_line(text: str) -> str:
    lines = text.splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if "RuntimeError" in stripped or "Traceback" in stripped or "ERROR" in stripped:
            return stripped
    return ""


def _parse_metrics(log_text: str) -> Dict[str, Any]:
    infer_s, motion_s = _last_inference_time(log_text)
    return {
        "l2": _last_float(r"l2 loss:\s*([0-9.eE+\-]+)", log_text),
        "lvel": _last_float(r"lvel loss:\s*([0-9.eE+\-]+)", log_text),
        "fid": _last_float(r"fid score:\s*([0-9.eE+\-]+)", log_text),
        "align": _last_float(r"align score:\s*([0-9.eE+\-]+)", log_text),
        "l1div": _last_float(r"l1div score:\s*([0-9.eE+\-]+)", log_text),
        "infer_seconds": infer_s,
        "motion_seconds": motion_s,
    }


def _run_one(
    root_dir: str,
    python_exec: str,
    config: str,
    speakers: List[int],
    candidate: Dict[str, Any],
    out_dir: str,
) -> Dict[str, Any]:
    name = candidate["name"]
    ckpt = candidate["ckpt"]
    extra_args = candidate["extra_args"]
    notes = candidate["notes"]

    log_path = os.path.join(out_dir, "%s.log" % name)
    cmd = [
        python_exec,
        "train.py",
        "--test_state",
        "--config",
        config,
        "--load_ckpt",
        ckpt,
        "--training_speakers",
    ] + [str(s) for s in speakers] + extra_args + ["--notes", notes, "--stat", "ts"]

    print("[RUN] %s" % name)
    print("      %s" % " ".join(shlex.quote(x) for x in cmd))

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("COMMAND: %s\n\n" % (" ".join(shlex.quote(x) for x in cmd)))
        proc = subprocess.run(cmd, cwd=root_dir, stdout=f, stderr=subprocess.STDOUT, text=True)
        wall_s = time.time() - t0
        f.write("\n[EXIT_CODE] %d\n" % proc.returncode)
        f.write("[WALL_SECONDS] %.3f\n" % wall_s)

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        log_text = f.read()

    metrics = _parse_metrics(log_text)

    result: Dict[str, Any] = {
        "name": name,
        "ckpt": ckpt,
        "extra_args": extra_args,
        "notes": notes,
        "returncode": proc.returncode,
        "wall_seconds": round(time.time() - t0, 3),
        "log_path": log_path,
        "status": "ok" if (proc.returncode == 0 and metrics.get("fid") is not None) else "failed",
        "error": "",
    }
    result.update(metrics)

    if result["status"] != "ok":
        result["error"] = _extract_error_line(log_text)

    return result


def _format_num(x: Any, digits: int = 6) -> str:
    if x is None:
        return "NA"
    if isinstance(x, int):
        return str(x)
    return ("%%.%df" % digits) % float(x)


def _avg(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def _write_csv(path: str, results: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "notes",
        "status",
        "fid",
        "align",
        "l1div",
        "l2",
        "lvel",
        "infer_seconds",
        "motion_seconds",
        "returncode",
        "wall_seconds",
        "ckpt",
        "log_path",
        "error",
        "extra_args",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fieldnames}
            row["extra_args"] = " ".join(r.get("extra_args", []))
            writer.writerow(row)


def _write_summary(
    path: str,
    config: str,
    speakers: List[int],
    results: List[Dict[str, Any]],
    top_k: int,
    baseline_name: Optional[str],
) -> None:
    ok = [r for r in results if r.get("status") == "ok" and r.get("fid") is not None]
    ok_sorted = sorted(ok, key=lambda x: float(x["fid"]))
    top = ok_sorted[: max(0, top_k)]

    baseline = None
    if baseline_name:
        for r in ok_sorted:
            if r["name"] == baseline_name:
                baseline = r
                break

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Fair DuoGesture-style Evaluation Summary\n\n")
        f.write("Date: %s\n\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        f.write("## Protocol\n\n")
        f.write("- Entry: train.py --test_state\n")
        f.write("- Config: %s\n" % config)
        f.write("- training_speakers: %s\n" % speakers)
        f.write("- Metrics parsed from trainer output: l2, lvel, fid, align, l1div, inference time\n\n")

        f.write("## All Results\n\n")
        f.write("| model | status | fid | align | l1div | l2 | lvel | infer_s | motion_s |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in ok_sorted:
            f.write(
                "| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n"
                % (
                    r["name"],
                    r["status"],
                    _format_num(r.get("fid"), 6),
                    _format_num(r.get("align"), 6),
                    _format_num(r.get("l1div"), 6),
                    _format_num(r.get("l2"), 8),
                    _format_num(r.get("lvel"), 8),
                    _format_num(r.get("infer_seconds"), 0),
                    _format_num(r.get("motion_seconds"), 0),
                )
            )
        failed = [r for r in results if r.get("status") != "ok"]
        for r in failed:
            f.write(
                "| %s | %s | NA | NA | NA | NA | NA | NA | NA |\n"
                % (r["name"], r["status"])
            )

        f.write("\n## Top-%d by FID\n\n" % top_k)
        if not top:
            f.write("No successful runs.\n")
        else:
            for i, r in enumerate(top, 1):
                f.write(
                    "%d. %s  (fid=%s, align=%s, l1div=%s, l2=%s, lvel=%s)\n"
                    % (
                        i,
                        r["name"],
                        _format_num(r.get("fid"), 6),
                        _format_num(r.get("align"), 6),
                        _format_num(r.get("l1div"), 6),
                        _format_num(r.get("l2"), 8),
                        _format_num(r.get("lvel"), 8),
                    )
                )

            f.write("\nTop-%d averages:\n\n" % top_k)
            f.write("- avg_fid: %s\n" % _format_num(_avg([r.get("fid") for r in top]), 6))
            f.write("- avg_align: %s\n" % _format_num(_avg([r.get("align") for r in top]), 6))
            f.write("- avg_l1div: %s\n" % _format_num(_avg([r.get("l1div") for r in top]), 6))
            f.write("- avg_l2: %s\n" % _format_num(_avg([r.get("l2") for r in top]), 8))
            f.write("- avg_lvel: %s\n" % _format_num(_avg([r.get("lvel") for r in top]), 8))

        if baseline is not None and top:
            best = top[0]
            delta = float(best["fid"]) - float(baseline["fid"])
            pct = 100.0 * (abs(delta) / float(baseline["fid"]))
            direction = "better" if delta < 0 else "worse"
            f.write("\n## Best vs Baseline\n\n")
            f.write(
                "Best (%s) vs baseline (%s): delta_fid=%s (%s, %.2f%%)\n"
                % (
                    best["name"],
                    baseline["name"],
                    _format_num(delta, 6),
                    direction,
                    pct,
                )
            )

        f.write("\n## Logs\n\n")
        for r in results:
            f.write("- %s: %s\n" % (r["name"], r["log_path"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Fair DuoGesture-style evaluator")
    parser.add_argument("--config", required=True, help="Config file used in --test_state")
    parser.add_argument("--candidates", required=True, help="JSON list of candidates")
    parser.add_argument(
        "--training_speakers",
        type=int,
        nargs="+",
        default=[2],
        help="Speaker IDs used for evaluation filtering",
    )
    parser.add_argument("--top_k", type=int, default=3, help="Top-k models for analysis by FID")
    parser.add_argument("--baseline_name", default="release_base", help="Baseline model name for delta report")
    parser.add_argument("--python_exec", default=sys.executable, help="Python executable for train.py")
    parser.add_argument("--out_dir", default=None, help="Output directory for logs and reports")
    parser.add_argument("--root_dir", default=".", help="Project root where train.py lives")

    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    config = args.config
    if not os.path.isabs(config):
        config = os.path.join(root_dir, config)

    candidates_path = args.candidates
    if not os.path.isabs(candidates_path):
        candidates_path = os.path.join(root_dir, candidates_path)

    if not os.path.isfile(config):
        raise SystemExit("Config not found: %s" % config)
    if not os.path.isfile(candidates_path):
        raise SystemExit("Candidates file not found: %s" % candidates_path)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(root_dir, "outputs", "findings", "fair_eval_%s" % stamp)
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(root_dir, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    candidates = _load_candidates(candidates_path)

    print("=" * 72)
    print("DuoGesture fair evaluator")
    print("=" * 72)
    print("root_dir:", root_dir)
    print("config:", config)
    print("candidates:", candidates_path)
    print("out_dir:", out_dir)
    print("training_speakers:", args.training_speakers)
    print("python_exec:", args.python_exec)
    print("models:", [c["name"] for c in candidates])
    print("=" * 72)

    results: List[Dict[str, Any]] = []
    for c in candidates:
        results.append(
            _run_one(
                root_dir=root_dir,
                python_exec=args.python_exec,
                config=config,
                speakers=args.training_speakers,
                candidate=c,
                out_dir=out_dir,
            )
        )

    results_json = os.path.join(out_dir, "results.json")
    results_csv = os.path.join(out_dir, "results.csv")
    summary_md = os.path.join(out_dir, "summary.md")

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _write_csv(results_csv, results)
    _write_summary(
        path=summary_md,
        config=config,
        speakers=args.training_speakers,
        results=results,
        top_k=args.top_k,
        baseline_name=args.baseline_name,
    )

    ok_count = sum(1 for r in results if r.get("status") == "ok")
    print("Done. Successful runs: %d/%d" % (ok_count, len(results)))
    print("results.json:", results_json)
    print("results.csv:", results_csv)
    print("summary.md:", summary_md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
