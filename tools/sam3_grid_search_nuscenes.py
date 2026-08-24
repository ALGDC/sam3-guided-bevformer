#!/usr/bin/env python
"""Run a compact SAM3/BEVFormer post-process grid search on nuScenes."""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_csv_floats(text):
    return [float(x) for x in text.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default="/122090720/datasets/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--bev-results", required=True)
    parser.add_argument("--sam3-results", required=True)
    parser.add_argument("--out-dir", default="work_dirs/sam3_nuscenes_mini/grid_search")
    parser.add_argument("--min-ious", default="0.05,0.10")
    parser.add_argument("--boosts", default="0.20,0.30,0.40")
    parser.add_argument("--no-match-factors", default="0.70,0.90")
    parser.add_argument("--multi-camera-weights", default="0.15,0.25")
    parser.add_argument("--min-sam-scores", default="0.0,0.35")
    parser.add_argument("--min-mask-box-coverages", default="0.0,0.05")
    parser.add_argument("--class-boosts-json", default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    return parser.parse_args()


def read_metric(path):
    metric_path = path / "eval" / "metrics_summary.json"
    if not metric_path.exists():
        return {}
    with metric_path.open("r") as f:
        data = json.load(f)
    return {
        "nd_score": data.get("nd_score"),
        "mean_ap": data.get("mean_ap"),
        "trans_err": data.get("tp_errors", {}).get("trans_err"),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = itertools.product(
        parse_csv_floats(args.min_ious),
        parse_csv_floats(args.boosts),
        parse_csv_floats(args.no_match_factors),
        parse_csv_floats(args.multi_camera_weights),
        parse_csv_floats(args.min_sam_scores),
        parse_csv_floats(args.min_mask_box_coverages),
    )

    rows = []
    for run_idx, (min_iou, boost, no_match, mc_weight, min_sam_score, min_mask_cov) in enumerate(combos):
        if args.max_runs is not None and run_idx >= args.max_runs:
            break

        run_name = (
            f"miou_{min_iou:.2f}_boost_{boost:.2f}_nmf_{no_match:.2f}_"
            f"mcw_{mc_weight:.2f}_ss_{min_sam_score:.2f}_mcov_{min_mask_cov:.2f}"
        )
        run_dir = out_dir / run_name
        result_path = run_dir / "results_nusc.json"
        debug_path = run_dir / "debug.json"
        eval_dir = run_dir / "eval"
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "tools/sam3_rescore_nuscenes.py",
            "--dataroot",
            args.dataroot,
            "--version",
            args.version,
            "--bev-results",
            args.bev_results,
            "--sam3-results",
            args.sam3_results,
            "--output",
            str(result_path),
            "--debug-output",
            str(debug_path),
            "--min-iou",
            str(min_iou),
            "--boost",
            str(boost),
            "--no-match-factor",
            str(no_match),
            "--vehicle-compatible",
            "--multi-camera",
            "--multi-camera-weight",
            str(mc_weight),
            "--mask-coverage",
            "--min-sam-score",
            str(min_sam_score),
            "--min-mask-box-coverage",
            str(min_mask_cov),
        ]
        if args.class_boosts_json:
            cmd += ["--class-boosts", Path(args.class_boosts_json).read_text().strip()]
        if args.eval:
            cmd += ["--eval", "--eval-output-dir", str(eval_dir)]

        print(f"[{run_idx}] {run_name}", flush=True)
        subprocess.run(cmd, check=True)

        row = {
            "run": run_name,
            "result": str(result_path),
            "debug": str(debug_path),
            "min_iou": min_iou,
            "boost": boost,
            "no_match_factor": no_match,
            "multi_camera_weight": mc_weight,
            "min_sam_score": min_sam_score,
            "min_mask_box_coverage": min_mask_cov,
        }
        row.update(read_metric(run_dir))
        rows.append(row)
        with (out_dir / "summary.json").open("w") as f:
            json.dump(rows, f, indent=2)

    ranked = sorted(rows, key=lambda x: -1.0 if x.get("nd_score") is None else x["nd_score"], reverse=True)
    with (out_dir / "summary_ranked.json").open("w") as f:
        json.dump(ranked, f, indent=2)
    if ranked:
        print("Best:", json.dumps(ranked[0], indent=2))


if __name__ == "__main__":
    main()
