#!/usr/bin/env python
"""Re-score nuScenes 3D detection results with SAM3 2D box support."""

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pyquaternion import Quaternion


CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default="/122090720/datasets/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument(
        "--bev-results",
        default="work_dirs/bevformer_base_nuscenes_mini/bev_weight_mini_detection/pts_bbox/results_nusc.json",
    )
    parser.add_argument(
        "--sam3-results",
        default="work_dirs/sam3_nuscenes_mini/sam3_2d_detections.json",
    )
    parser.add_argument(
        "--output",
        default="work_dirs/sam3_nuscenes_mini/results_nusc_sam3_rescore.json",
    )
    parser.add_argument(
        "--debug-output",
        default="work_dirs/sam3_nuscenes_mini/sam3_rescore_debug.json",
    )
    parser.add_argument("--min-iou", type=float, default=0.10)
    parser.add_argument("--boost", type=float, default=0.30)
    parser.add_argument(
        "--class-boosts",
        default=None,
        help='Optional JSON dict for per-class boost, e.g. \'{"car":0.3,"bicycle":0.6}\'.',
    )
    parser.add_argument("--no-match-factor", type=float, default=0.70)
    parser.add_argument(
        "--vehicle-compatible",
        action="store_true",
        help="Allow vehicle classes to support each other during 2D matching.",
    )
    parser.add_argument(
        "--multi-camera",
        action="store_true",
        help="Accumulate support from multiple cameras instead of only using one best view.",
    )
    parser.add_argument(
        "--multi-camera-weight",
        type=float,
        default=0.25,
        help="Weight for additional camera supports when --multi-camera is enabled.",
    )
    parser.add_argument(
        "--mask-coverage",
        action="store_true",
        help="Use SAM3 mask coverage when mask_rle is present in SAM3 detections.",
    )
    parser.add_argument("--iou-weight", type=float, default=0.60)
    parser.add_argument("--coverage-weight", type=float, default=0.40)
    parser.add_argument(
        "--filter-no-match",
        action="store_true",
        help="Drop boxes that have no SAM3 2D support above --min-iou.",
    )
    parser.add_argument("--min-sam-score", type=float, default=0.0)
    parser.add_argument("--min-sam-area", type=float, default=0.0)
    parser.add_argument("--max-sam-area-frac", type=float, default=1.0)
    parser.add_argument("--min-sam-aspect", type=float, default=0.0)
    parser.add_argument("--max-sam-aspect", type=float, default=1.0e9)
    parser.add_argument(
        "--min-mask-box-coverage",
        type=float,
        default=0.0,
        help="Drop SAM3 masks whose foreground pixels cover too little of their own box.",
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--eval-output-dir",
        default="work_dirs/sam3_nuscenes_mini/eval_rescore",
    )
    return parser.parse_args()


def box_iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def labels_compatible(det_label: str, sam_label: str, vehicle_compatible: bool) -> bool:
    if det_label == sam_label:
        return True
    if vehicle_compatible and det_label in VEHICLE_CLASSES and sam_label in VEHICLE_CLASSES:
        return True
    return False


def mask_coverage_in_box(mask_rle: dict, box: List[float]) -> Optional[float]:
    if not mask_rle:
        return None
    try:
        from pycocotools import mask as mask_utils
    except Exception:
        return None

    rle = dict(mask_rle)
    counts = rle.get("counts")
    if isinstance(counts, str):
        rle["counts"] = counts.encode("ascii")
    mask = mask_utils.decode(rle)
    x1, y1, x2, y2 = box
    h, w = mask.shape[:2]
    x1 = int(max(0, min(w, np.floor(x1))))
    y1 = int(max(0, min(h, np.floor(y1))))
    x2 = int(max(0, min(w, np.ceil(x2))))
    y2 = int(max(0, min(h, np.ceil(y2))))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = mask[y1:y2, x1:x2] > 0
    return float(crop.mean()) if crop.size else 0.0


def sam3_detection_passes_quality(det: dict, args) -> bool:
    if float(det.get("score", 1.0)) < args.min_sam_score:
        return False

    x1, y1, x2, y2 = det["bbox"]
    width = max(0.0, float(x2) - float(x1))
    height = max(0.0, float(y2) - float(y1))
    area = width * height
    if area < args.min_sam_area:
        return False

    image_area = None
    mask_rle = det.get("mask_rle")
    if mask_rle and "size" in mask_rle:
        image_area = float(mask_rle["size"][0] * mask_rle["size"][1])
    if image_area and image_area > 0 and area / image_area > args.max_sam_area_frac:
        return False

    if width <= 0.0 or height <= 0.0:
        return False
    aspect = width / height
    if aspect < args.min_sam_aspect or aspect > args.max_sam_aspect:
        return False

    if args.min_mask_box_coverage > 0.0 and mask_rle:
        coverage = mask_coverage_in_box(mask_rle, det["bbox"])
        if coverage is not None and coverage < args.min_mask_box_coverage:
            return False

    return True


def filter_sam3_detections(sam3_by_sample: Dict[str, Dict[str, List[dict]]], args):
    filtered: Dict[str, Dict[str, List[dict]]] = {}
    kept = 0
    dropped = 0
    for sample_token, cams in sam3_by_sample.items():
        out_cams = {}
        for cam, dets in cams.items():
            out_dets = []
            for det in dets:
                if sam3_detection_passes_quality(det, args):
                    out_dets.append(det)
                    kept += 1
                else:
                    dropped += 1
            out_cams[cam] = out_dets
        filtered[sample_token] = out_cams
    return filtered, {"kept": kept, "dropped": dropped}


def get_image_size(dataroot: str, sample_data: dict) -> Tuple[int, int]:
    width = sample_data.get("width", None)
    height = sample_data.get("height", None)
    if width and height:
        return int(width), int(height)
    image_path = Path(dataroot) / sample_data["filename"]
    with Image.open(image_path) as image:
        return image.size


def project_detection_to_camera(nusc, dataroot: str, det: dict, cam_sample_data: dict) -> Optional[List[float]]:
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import view_points

    box = Box(
        center=det["translation"],
        size=det["size"],
        orientation=Quaternion(det["rotation"]),
        name=det.get("detection_name", ""),
        score=det.get("detection_score", 0.0),
    )

    ego_pose = nusc.get("ego_pose", cam_sample_data["ego_pose_token"])
    box.translate(-np.array(ego_pose["translation"]))
    box.rotate(Quaternion(ego_pose["rotation"]).inverse)

    calibrated_sensor = nusc.get("calibrated_sensor", cam_sample_data["calibrated_sensor_token"])
    box.translate(-np.array(calibrated_sensor["translation"]))
    box.rotate(Quaternion(calibrated_sensor["rotation"]).inverse)

    corners_3d = box.corners()
    depths = corners_3d[2, :]
    if np.any(depths <= 0.1):
        return None

    camera_intrinsic = np.array(calibrated_sensor["camera_intrinsic"])
    corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]
    width, height = get_image_size(dataroot, cam_sample_data)

    x1 = float(np.min(corners_2d[0, :]))
    y1 = float(np.min(corners_2d[1, :]))
    x2 = float(np.max(corners_2d[0, :]))
    y2 = float(np.max(corners_2d[1, :]))

    x1 = max(0.0, min(x1, width - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    x2 = max(0.0, min(x2, width - 1.0))
    y2 = max(0.0, min(y2, height - 1.0))
    if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
        return None
    return [x1, y1, x2, y2]


def find_sam3_support(
    nusc,
    dataroot: str,
    det: dict,
    sample_token: str,
    sam3_by_sample: Dict[str, Dict[str, List[dict]]],
    min_iou: float,
    vehicle_compatible: bool = False,
    multi_camera: bool = False,
    multi_camera_weight: float = 0.25,
    mask_coverage: bool = False,
    iou_weight: float = 0.60,
    coverage_weight: float = 0.40,
) -> dict:
    sample = nusc.get("sample", sample_token)
    label = det["detection_name"]
    best = {
        "support": 0.0,
        "best_iou": 0.0,
        "best_sam3_score": 0.0,
        "camera": None,
        "projected_bbox": None,
        "sam3_bbox": None,
        "coverage": None,
        "cameras_supported": 0,
        "support_sum": 0.0,
    }
    per_camera_best = []

    for cam in CAMERAS:
        cam_data = nusc.get("sample_data", sample["data"][cam])
        projected = project_detection_to_camera(nusc, dataroot, det, cam_data)
        if projected is None:
            continue

        cam_best = None
        for sam_det in sam3_by_sample.get(sample_token, {}).get(cam, []):
            if not labels_compatible(label, sam_det.get("label"), vehicle_compatible):
                continue
            iou = box_iou_xyxy(projected, sam_det["bbox"])
            if iou < min_iou:
                continue
            sam_score = float(sam_det.get("score", 1.0))
            coverage = None
            spatial_score = iou
            if mask_coverage and "mask_rle" in sam_det:
                coverage = mask_coverage_in_box(sam_det["mask_rle"], projected)
                if coverage is not None:
                    spatial_score = iou_weight * iou + coverage_weight * coverage
            support = spatial_score * sam_score
            candidate = {
                "support": support,
                "best_iou": iou,
                "best_sam3_score": sam_score,
                "camera": cam,
                "projected_bbox": projected,
                "sam3_bbox": sam_det["bbox"],
                "coverage": coverage,
                "sam3_label": sam_det.get("label"),
            }
            if cam_best is None or support > cam_best["support"]:
                cam_best = candidate
            if support > best["support"]:
                best = candidate
        if cam_best is not None:
            per_camera_best.append(cam_best)

    if multi_camera and per_camera_best:
        per_camera_best = sorted(per_camera_best, key=lambda x: x["support"], reverse=True)
        support_sum = sum(x["support"] for x in per_camera_best)
        combined = per_camera_best[0]["support"] + multi_camera_weight * sum(
            x["support"] for x in per_camera_best[1:]
        )
        best = dict(per_camera_best[0])
        best["support"] = min(1.0, combined)
        best["support_sum"] = support_sum
        best["cameras_supported"] = len(per_camera_best)
    elif per_camera_best:
        best["support_sum"] = sum(x["support"] for x in per_camera_best)
        best["cameras_supported"] = len(per_camera_best)

    return best


def rescore_detection(
    det: dict,
    support: float,
    boost: float,
    no_match_factor: float,
    class_boosts: Optional[Dict[str, float]] = None,
) -> float:
    original = float(det["detection_score"])
    det_boost = boost
    if class_boosts and det.get("detection_name") in class_boosts:
        det_boost = float(class_boosts[det["detection_name"]])
    if support > 0.0:
        return min(1.0, original * (1.0 + det_boost * support))
    return max(0.0, original * no_match_factor)


def evaluate_nuscenes(result_path: str, dataroot: str, version: str, output_dir: str):
    from nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    eval_set_map = {
        "v1.0-mini": "mini_val",
        "v1.0-trainval": "val",
        "v1.0-test": "test",
    }
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
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
    with open(args.sam3_results, "r") as f:
        sam3_results = json.load(f)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    sam3_by_sample, sam3_quality = filter_sam3_detections(sam3_results["detections"], args)
    class_boosts = json.loads(args.class_boosts) if args.class_boosts else None

    rescored = copy.deepcopy(bev_results)
    debug = {
        "meta": {
            "bev_results": args.bev_results,
            "sam3_results": args.sam3_results,
            "min_iou": args.min_iou,
            "boost": args.boost,
            "class_boosts": class_boosts,
            "no_match_factor": args.no_match_factor,
            "filter_no_match": args.filter_no_match,
            "vehicle_compatible": args.vehicle_compatible,
            "multi_camera": args.multi_camera,
            "multi_camera_weight": args.multi_camera_weight,
            "mask_coverage": args.mask_coverage,
            "iou_weight": args.iou_weight,
            "coverage_weight": args.coverage_weight,
            "min_sam_score": args.min_sam_score,
            "min_sam_area": args.min_sam_area,
            "max_sam_area_frac": args.max_sam_area_frac,
            "min_sam_aspect": args.min_sam_aspect,
            "max_sam_aspect": args.max_sam_aspect,
            "min_mask_box_coverage": args.min_mask_box_coverage,
            "sam3_quality": sam3_quality,
        },
        "detections": {},
    }

    kept = 0
    dropped = 0
    supported = 0

    for sample_token, detections in bev_results["results"].items():
        new_detections = []
        sample_debug = []
        for det_idx, det in enumerate(detections):
            support = find_sam3_support(
                nusc=nusc,
                dataroot=args.dataroot,
                det=det,
                sample_token=sample_token,
                sam3_by_sample=sam3_by_sample,
                min_iou=args.min_iou,
                vehicle_compatible=args.vehicle_compatible,
                multi_camera=args.multi_camera,
                multi_camera_weight=args.multi_camera_weight,
                mask_coverage=args.mask_coverage,
                iou_weight=args.iou_weight,
                coverage_weight=args.coverage_weight,
            )
            original_score = float(det["detection_score"])
            new_score = rescore_detection(
                det,
                support["support"],
                args.boost,
                args.no_match_factor,
                class_boosts=class_boosts,
            )

            if support["support"] <= 0.0 and args.filter_no_match:
                dropped += 1
                keep = False
            else:
                keep = True

            sample_debug.append(
                {
                    "index": det_idx,
                    "name": det["detection_name"],
                    "original_score": original_score,
                    "new_score": new_score,
                    **support,
                    "kept": keep,
                }
            )

            if keep:
                new_det = copy.deepcopy(det)
                new_det["detection_score"] = new_score
                new_detections.append(new_det)
                kept += 1
                if support["support"] > 0.0:
                    supported += 1

        rescored["results"][sample_token] = new_detections
        debug["detections"][sample_token] = sample_debug

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(rescored, f)

    debug_path = Path(args.debug_output)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with debug_path.open("w") as f:
        json.dump(debug, f)

    total = kept + dropped
    ratio = supported / kept if kept else math.nan
    print(f"Saved rescored result JSON: {output_path}")
    print(f"Saved debug JSON: {debug_path}")
    print(f"Detections total={total}, kept={kept}, dropped={dropped}, supported={supported}, support_ratio={ratio:.4f}")

    if args.eval:
        evaluate_nuscenes(str(output_path), args.dataroot, args.version, args.eval_output_dir)


if __name__ == "__main__":
    main()
