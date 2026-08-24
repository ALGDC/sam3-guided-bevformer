#!/usr/bin/env python
"""Smoke-test 15D query-only SAM mask-aware proposal encoding."""

import argparse

import mmcv
import numpy as np
import torch
from PIL import Image

import projects.mmdet3d_plugin  # noqa: F401
from projects.mmdet3d_plugin.datasets.pipelines.loading import CAMERAS, LoadSAM3Proposals
from projects.mmdet3d_plugin.models.utils.sam_query_encoder import SAMProposalQueryEncoder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ann-file",
        default="/122090720/datasets/nuscenes/nuscenes_infos_temporal_val.pkl",
    )
    parser.add_argument(
        "--dataroot",
        default="/122090720/datasets/nuscenes",
    )
    parser.add_argument(
        "--sam3-results",
        default="/122090720/bevformer/BEVFormer/work_dirs/sam3_nuscenes_mini/sam3_2d_detections_with_masks.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    infos = mmcv.load(args.ann_file)
    infos = infos["infos"] if isinstance(infos, dict) and "infos" in infos else infos

    loader = LoadSAM3Proposals(
        sam3_results=args.sam3_results,
        max_proposals=50,
        min_score=0.30,
        proposal_dim=15,
    )

    sample = None
    for info in infos:
        cam_intrinsic = []
        lidar2cam = []
        ori_shape = []
        for cam_name in CAMERAS:
            cam = info["cams"][cam_name]
            img_path = cam["data_path"].replace("./data/nuscenes", args.dataroot)
            img = Image.open(img_path)
            w, h = img.size
            ori_shape.append((h, w, 3))
            cam_intrinsic.append(np.asarray(cam["cam_intrinsic"], dtype=np.float32))
            rotation = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float32)
            translation = np.asarray(cam["sensor2lidar_translation"], dtype=np.float32)
            cam2lidar = np.eye(4, dtype=np.float32)
            cam2lidar[:3, :3] = rotation
            cam2lidar[:3, 3] = translation
            lidar2cam.append(np.linalg.inv(cam2lidar))
        results = dict(
            sample_idx=info["token"],
            ori_shape=ori_shape,
            img_shape=ori_shape,
            cam_intrinsic=cam_intrinsic,
            lidar2cam=lidar2cam,
        )
        results = loader(results)
        if results["sam_proposals"].shape[0] > 0:
            sample = results["sam_proposals"]
            break

    if sample is None:
        raise RuntimeError("No non-empty 15D SAM proposal sample found.")

    print("sam_proposals", sample.shape)
    print("first_row", sample[0])

    encoder = SAMProposalQueryEncoder(embed_dims=256)
    proposals = torch.from_numpy(sample).unsqueeze(0)
    mask = torch.ones((1, proposals.size(1)), dtype=torch.bool)
    out = encoder(proposals, mask)
    print("query_delta", tuple(out.shape), bool(torch.isfinite(out).all()))


if __name__ == "__main__":
    main()
