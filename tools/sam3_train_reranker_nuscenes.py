#!/usr/bin/env python
"""Train/apply a small SAM3-aware score re-ranker for nuScenes result JSONs.

This is intended for quick mini-set experiments. When train and eval use the
same mini split, treat the metric as an upper-bound/ablation signal rather than
a publishable validation number.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default="/122090720/datasets/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--bev-results", required=True)
    parser.add_argument("--debug-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-output", default=None)
    parser.add_argument("--distance-thr", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1.0e-3)
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=1.0,
        help="Final score = (1-alpha) * input detection_score + alpha * learned score.",
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-output-dir", default=None)
    return parser.parse_args()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def build_gt_index(nusc):
    gt_by_sample: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for sample in nusc.sample:
        per_class = {name: [] for name in CLASSES}
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            det_name = getattr(nusc, "get_box", None)
            category = ann["category_name"]
            # nuScenes detection names are the suffix for these categories.
            name = category.split(".")[-1]
            if category == "human.pedestrian.adult" or category.startswith("human.pedestrian"):
                name = "pedestrian"
            elif category == "vehicle.construction":
                name = "construction_vehicle"
            elif category == "movable_object.barrier":
                name = "barrier"
            elif category == "movable_object.trafficcone":
                name = "traffic_cone"
            if name in per_class:
                per_class[name].append(np.asarray(ann["translation"][:2], dtype=np.float32))
        gt_by_sample[sample["token"]] = per_class
    return gt_by_sample


def make_feature(det, dbg):
    size = det.get("size", [0.0, 0.0, 0.0])
    area = 0.0
    proj = dbg.get("projected_bbox")
    if proj:
        area = max(0.0, proj[2] - proj[0]) * max(0.0, proj[3] - proj[1])
    coverage = dbg.get("coverage")
    if coverage is None:
        coverage = 0.0
    class_id = CLASSES.index(det["detection_name"]) if det["detection_name"] in CLASSES else -1
    one_hot = [1.0 if i == class_id else 0.0 for i in range(len(CLASSES))]
    return np.asarray(
        [
            float(det["detection_score"]),
            float(dbg.get("new_score", det["detection_score"])),
            float(dbg.get("support", 0.0)),
            float(dbg.get("support_sum", 0.0)),
            float(dbg.get("best_iou", 0.0)),
            float(dbg.get("best_sam3_score", 0.0)),
            float(coverage),
            float(dbg.get("cameras_supported", 0)),
            math.log1p(area),
            float(size[0]),
            float(size[1]),
            float(size[2]),
        ]
        + one_hot,
        dtype=np.float32,
    )


def label_detection(det, gt_by_sample, sample_token, distance_thr):
    label = det["detection_name"]
    center = np.asarray(det["translation"][:2], dtype=np.float32)
    gts = gt_by_sample.get(sample_token, {}).get(label, [])
    if not gts:
        return 0.0
    min_dist = min(float(np.linalg.norm(center - gt)) for gt in gts)
    return 1.0 if min_dist <= distance_thr else 0.0


def collect_examples(bev_results, debug, gt_by_sample, distance_thr):
    feats = []
    labels = []
    refs = []
    for sample_token, detections in bev_results["results"].items():
        dbg_items = debug["detections"].get(sample_token, [])
        dbg_by_idx = {int(item["index"]): item for item in dbg_items}
        for idx, det in enumerate(detections):
            dbg = dbg_by_idx.get(idx, {})
            feats.append(make_feature(det, dbg))
            labels.append(label_detection(det, gt_by_sample, sample_token, distance_thr))
            refs.append((sample_token, idx))
    return np.stack(feats), np.asarray(labels, dtype=np.float32), refs


def train_logistic(x, y, epochs, lr, l2):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1.0e-6] = 1.0
    xs = (x - mean) / std
    xs = np.concatenate([xs, np.ones((xs.shape[0], 1), dtype=np.float32)], axis=1)
    w = np.zeros((xs.shape[1],), dtype=np.float32)
    pos_weight = float((len(y) - y.sum()) / max(y.sum(), 1.0))
    weights = np.where(y > 0.5, pos_weight, 1.0).astype(np.float32)
    for _ in range(epochs):
        pred = sigmoid(xs @ w)
        grad = xs.T @ ((pred - y) * weights) / len(y)
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad
    return {"mean": mean.tolist(), "std": std.tolist(), "weights": w.tolist()}


def predict(model, x):
    mean = np.asarray(model["mean"], dtype=np.float32)
    std = np.asarray(model["std"], dtype=np.float32)
    w = np.asarray(model["weights"], dtype=np.float32)
    xs = (x - mean) / std
    xs = np.concatenate([xs, np.ones((xs.shape[0], 1), dtype=np.float32)], axis=1)
    return sigmoid(xs @ w)


def evaluate_nuscenes(result_path: str, dataroot: str, version: str, output_dir: str):
    from nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    eval_set_map = {
        "v1.0-mini": "mini_val",
        "v1.0-trainval": "val",
        "v1.0-test": "test",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    evaluator = NuScenesEval(
        nusc,
        config=config_factory("detection_cvpr_2019"),
        result_path=result_path,
        eval_set=eval_set_map[version],
        output_dir=str(output),
        verbose=True,
    )
    evaluator.main(plot_examples=0, render_curves=False)


def main():
    args = parse_args()
    from nuscenes import NuScenes

    with open(args.bev_results, "r") as f:
        bev_results = json.load(f)
    with open(args.debug_json, "r") as f:
        debug = json.load(f)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    gt_by_sample = build_gt_index(nusc)
    x, y, refs = collect_examples(bev_results, debug, gt_by_sample, args.distance_thr)
    model = train_logistic(x, y, args.epochs, args.lr, args.l2)
    scores = predict(model, x)

    reranked = json.loads(json.dumps(bev_results))
    alpha = max(0.0, min(1.0, args.blend_alpha))
    for score, (sample_token, idx) in zip(scores, refs):
        det = reranked["results"][sample_token][idx]
        input_score = float(det["detection_score"])
        det["detection_score"] = float((1.0 - alpha) * input_score + alpha * score)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(reranked, f)

    if args.model_output:
        payload = {
            "model": model,
            "feature_names": [
                "orig_score",
                "rescored_score",
                "support",
                "support_sum",
                "best_iou",
                "best_sam3_score",
                "coverage",
                "cameras_supported",
                "log_projected_area",
                "size_x",
                "size_y",
                "size_z",
            ]
            + [f"class_{name}" for name in CLASSES],
            "positive_ratio": float(y.mean()),
            "num_examples": int(len(y)),
            "blend_alpha": alpha,
        }
        with open(args.model_output, "w") as f:
            json.dump(payload, f, indent=2)

    print(f"Saved reranked result JSON: {output}")
    print(f"Training examples={len(y)}, positive_ratio={y.mean():.4f}")

    if args.eval:
        eval_dir = args.eval_output_dir or str(output.parent / "eval")
        evaluate_nuscenes(str(output), args.dataroot, args.version, eval_dir)


if __name__ == "__main__":
    main()
