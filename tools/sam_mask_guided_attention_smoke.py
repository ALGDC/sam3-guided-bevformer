#!/usr/bin/env python
"""Smoke-test SAM mask-guided attention inputs and module wiring."""

import argparse

import mmcv
import numpy as np
import torch
from PIL import Image
from mmcv import Config

import projects.mmdet3d_plugin  # noqa: F401
from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import SpatialCrossAttention
from projects.mmdet3d_plugin.datasets.pipelines.loading import (
    CAMERAS,
    LoadSAM3MaskPriors,
    LoadSAM3Proposals,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="projects/configs/bevformer/bevformer-tiny-sam3-query-13d-maskattn.py",
    )
    parser.add_argument(
        "--dataroot",
        default="/122090720/datasets/nuscenes/",
    )
    parser.add_argument(
        "--ann-file",
        default="/122090720/datasets/nuscenes/nuscenes_infos_temporal_val.pkl",
    )
    parser.add_argument(
        "--sam3-results",
        default="work_dirs/sam3_nuscenes_mini/sam3_2d_detections_with_masks.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    infos = mmcv.load(args.ann_file)
    infos = infos["infos"] if isinstance(infos, dict) and "infos" in infos else infos

    proposal_loader = LoadSAM3Proposals(
        sam3_results=args.sam3_results,
        max_proposals=50,
        min_score=0.30,
        proposal_dim=13,
    )
    prior_loader = LoadSAM3MaskPriors(
        sam3_results=args.sam3_results,
        min_score=0.30,
        prior_size=(32, 56),
    )

    img_metas = None
    for info in infos:
        cam_intrinsic = []
        lidar2cam = []
        ori_shape = []
        for cam_name in CAMERAS:
            cam = info["cams"][cam_name]
            img_path = cam["data_path"].replace("./data/nuscenes", args.dataroot.rstrip("/"))
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
        results = proposal_loader(results)
        results = prior_loader(results)
        if results["sam_proposals"].shape[0] > 0:
            img_metas = results
            break

    if img_metas is None:
        raise RuntimeError("No non-empty SAM proposal sample found for smoke test.")

    sam_props = img_metas["sam_proposals"]
    sam_mask_priors = img_metas["sam_mask_priors"]
    print("sam_proposals", sam_props.shape)
    print("sam_mask_priors", sam_mask_priors.shape)

    attn = SpatialCrossAttention(
        embed_dims=256,
        num_cams=6,
        pc_range=cfg.point_cloud_range,
        use_sam_mask_attention=True,
        sam_mask_attention_scale=0.5,
        deformable_attention=dict(
            type="MSDeformableAttention3D",
            embed_dims=256,
            num_levels=1,
            num_points=8,
        ),
    )

    bs = 1
    max_len = 4
    depth = 4
    num_query = 8
    num_value = 16
    query = torch.randn(bs, num_query, 256)
    key = torch.randn(6, num_value, bs, 256)
    value = torch.randn(6, num_value, bs, 256)
    reference_points_cam = torch.rand(6, bs, num_query, depth, 2)
    bev_mask = torch.ones(6, bs, num_query, depth, dtype=torch.bool)
    spatial_shapes = torch.tensor([[4, 4]], dtype=torch.long)
    level_start_index = torch.tensor([0], dtype=torch.long)

    gate = attn._build_mask_attention_gate(
        reference_points_cam.permute(1, 0, 2, 3, 4),
        [img_metas],
        query.device,
        query.dtype,
    )
    print("mask_gate", None if gate is None else tuple(gate.shape), bool(torch.isfinite(gate).all()))

    out = attn(
        query=query,
        key=key,
        value=value,
        reference_points_cam=reference_points_cam,
        bev_mask=bev_mask,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        img_metas=[img_metas],
    )
    print("attn_out", tuple(out.shape), bool(torch.isfinite(out).all()))


if __name__ == "__main__":
    main()
