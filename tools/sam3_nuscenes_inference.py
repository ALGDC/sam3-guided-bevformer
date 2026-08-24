#!/usr/bin/env python
"""Run SAM3 text-prompt detections on nuScenes camera images.

The output is a lightweight JSON consumed by sam3_rescore_nuscenes.py:

{
  "meta": {...},
  "detections": {
    "<sample_token>": {
      "CAM_FRONT": [
        {"bbox": [x1, y1, x2, y2], "score": 0.9, "label": "car", ...}
      ]
    }
  }
}
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


DEFAULT_PROMPTS = {
    "car": ["car"],
    "truck": ["truck"],
    "construction_vehicle": ["construction vehicle"],
    "bus": ["bus"],
    "trailer": ["trailer"],
    "barrier": ["barrier"],
    "motorcycle": ["motorcycle"],
    "bicycle": ["bicycle"],
    "pedestrian": ["pedestrian", "person"],
    "traffic_cone": ["traffic cone"],
}

CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default="/122090720/datasets/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument(
        "--ann-file",
        default="/122090720/datasets/nuscenes/nuscenes_infos_temporal_val.pkl",
        help="BEVFormer nuScenes info pkl. This avoids requiring nuscenes-devkit in the SAM3 env.",
    )
    parser.add_argument(
        "--bev-results",
        default="work_dirs/bevformer_base_nuscenes_mini/bev_weight_mini_detection/pts_bbox/results_nusc.json",
        help="Use sample tokens from this BEVFormer result JSON.",
    )
    parser.add_argument("--sam3-repo", default="/122090720/sam3/sam3_repo")
    parser.add_argument("--checkpoint", default="/122090720/sam3/sam3_weights/sam3.pt")
    parser.add_argument(
        "--output",
        default="work_dirs/sam3_nuscenes_mini/sam3_2d_detections.json",
    )
    parser.add_argument("--score-thr", type=float, default=0.30)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save SAM3 masks as COCO RLE. Needed for mask coverage re-score.",
    )
    parser.add_argument(
        "--prompts-json",
        default=None,
        help="Optional JSON file mapping nuScenes class names to text prompts.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def load_sample_tokens(bev_results_path: Path) -> List[str]:
    with bev_results_path.open("r") as f:
        data = json.load(f)
    return list(data["results"].keys())


def load_nuscenes_infos(ann_file: Path) -> Dict[str, dict]:
    with ann_file.open("rb") as f:
        data = pickle.load(f)
    return {info["token"]: info for info in data["infos"]}


def to_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def clip_xyxy(box, width: int, height: int):
    x1, y1, x2, y2 = [to_float(v) for v in box]
    x1 = max(0.0, min(x1, width - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    x2 = max(0.0, min(x2, width - 1.0))
    y2 = max(0.0, min(y2, height - 1.0))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def nms_xyxy(dets: List[dict], iou_thr: float) -> List[dict]:
    if not dets:
        return dets
    try:
        from torchvision.ops import nms
    except Exception:
        return dets

    boxes = torch.tensor([d["bbox"] for d in dets], dtype=torch.float32)
    scores = torch.tensor([d["score"] for d in dets], dtype=torch.float32)
    keep = nms(boxes, scores, iou_thr).cpu().tolist()
    return [dets[i] for i in keep]


def encode_mask_rle(mask):
    try:
        from pycocotools import mask as mask_utils
    except Exception:
        return None

    mask_np = mask
    if isinstance(mask_np, torch.Tensor):
        mask_np = mask_np.detach().cpu().numpy()
    mask_np = np.asarray(mask_np)
    while mask_np.ndim > 2:
        mask_np = mask_np[0]
    mask_np = np.asfortranarray((mask_np > 0).astype(np.uint8))
    rle = mask_utils.encode(mask_np)
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def extract_output_boxes(output, label: str, prompt: str, width: int, height: int, score_thr: float, save_masks: bool):
    boxes = output.get("boxes", None)
    scores = output.get("scores", None)
    if boxes is None or len(boxes) == 0:
        return []
    masks = output.get("masks", None)

    detections = []
    for idx in range(len(boxes)):
        score = 1.0
        if scores is not None and len(scores) > idx:
            score = to_float(scores[idx])
        if score < score_thr:
            continue

        x1, y1, x2, y2 = clip_xyxy(boxes[idx], width, height)
        if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
            continue

        det = {
            "bbox": [x1, y1, x2, y2],
            "score": score,
            "label": label,
            "prompt": prompt,
        }
        if save_masks and masks is not None and len(masks) > idx:
            rle = encode_mask_rle(masks[idx])
            if rle is not None:
                det["mask_rle"] = rle
        detections.append(det)
    return detections


def load_prompts(path: str | None) -> Dict[str, List[str]]:
    if path is None:
        return DEFAULT_PROMPTS
    with open(path, "r") as f:
        prompts = json.load(f)
    return {label: list(values) for label, values in prompts.items()}


def resolve_image_path(dataroot: str, data_path: str) -> Path:
    """Resolve image paths stored by different nuScenes converters.

    Some info pkls store paths like:
      data/nuscenes/samples/CAM_FRONT/xxx.jpg
    while the local dataroot is already:
      /122090720/datasets/nuscenes

    In that case joining dataroot + data_path duplicates "data/nuscenes".
    """
    raw = Path(data_path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path(dataroot) / raw)
        parts = raw.parts
        if "samples" in parts:
            sample_idx = parts.index("samples")
            candidates.append(Path(dataroot).joinpath(*parts[sample_idx:]))
        if "sweeps" in parts:
            sweep_idx = parts.index("sweeps")
            candidates.append(Path(dataroot).joinpath(*parts[sweep_idx:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def iter_camera_images(infos_by_token: Dict[str, dict], sample_tokens: Iterable[str]):
    for sample_token in sample_tokens:
        info = infos_by_token.get(sample_token)
        if info is None:
            print(f"[WARN] sample token not found in ann file: {sample_token}")
            continue
        for cam in CAMERAS:
            cam_info = info["cams"].get(cam)
            if cam_info is None:
                print(f"[WARN] camera not found: sample={sample_token} cam={cam}")
                continue
            yield sample_token, cam, cam_info


def main():
    args = parse_args()
    sys.path.insert(0, args.sam3_repo)

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    prompts = load_prompts(args.prompts_json)
    sample_tokens = load_sample_tokens(Path(args.bev_results))
    if args.max_samples is not None:
        sample_tokens = sample_tokens[: args.max_samples]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading nuScenes infos: {args.ann_file}")
    infos_by_token = load_nuscenes_infos(Path(args.ann_file))

    print(f"Loading SAM3 checkpoint: {args.checkpoint}")
    model = build_sam3_image_model(checkpoint_path=args.checkpoint)
    processor = Sam3Processor(model)

    all_detections: Dict[str, Dict[str, List[dict]]] = {}
    camera_items = list(iter_camera_images(infos_by_token, sample_tokens))

    for sample_token, cam, cam_info in tqdm(camera_items, desc="SAM3 nuScenes"):
        image_path = resolve_image_path(args.dataroot, cam_info["data_path"])
        if not image_path.exists():
            print(f"[WARN] missing image: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        state = processor.set_image(image)

        per_label: Dict[str, List[dict]] = {}
        for label, prompt_list in prompts.items():
            per_label[label] = []
            for prompt in prompt_list:
                try:
                    output = processor.set_text_prompt(state=state, prompt=prompt)
                except Exception as exc:
                    print(f"[WARN] SAM3 failed: sample={sample_token} cam={cam} prompt={prompt}: {exc}")
                    continue
                per_label[label].extend(
                    extract_output_boxes(
                        output,
                        label,
                        prompt,
                        width,
                        height,
                        args.score_thr,
                        args.save_masks,
                    )
                )

        cam_dets: List[dict] = []
        for label, dets in per_label.items():
            cam_dets.extend(nms_xyxy(dets, args.nms_iou))

        for det in cam_dets:
            det["camera"] = cam
            det["image_path"] = str(image_path)

        all_detections.setdefault(sample_token, {})[cam] = cam_dets

    payload = {
        "meta": {
            "dataroot": args.dataroot,
            "version": args.version,
            "checkpoint": args.checkpoint,
            "score_thr": args.score_thr,
            "nms_iou": args.nms_iou,
            "save_masks": args.save_masks,
            "prompts": prompts,
            "num_samples": len(sample_tokens),
        },
        "detections": all_detections,
    }
    with output_path.open("w") as f:
        json.dump(payload, f)

    total = sum(len(v) for cams in all_detections.values() for v in cams.values())
    print(f"Saved SAM3 detections to: {output_path}")
    print(f"Total 2D detections: {total}")


if __name__ == "__main__":
    main()
