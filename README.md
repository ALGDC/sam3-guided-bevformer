# SAM3-Guided BEVFormer for Camera-Only 3D Detection

This repository is a BEVFormer-based research branch for camera-only 3D object detection with SAM3 proposal guidance on nuScenes.

The current best-performing branch in this repo uses:

- `13D SAM query initialization`
- `BEV prior injection`
- conservative prior settings:
  - `sam_bev_prior_scale=0.05`
  - `sam_bev_prior_radius=2.0`

In short, the method extends 2D SAM3 detections into geometry-aware proposal tokens and injects them into BEVFormer at two levels:

1. object-query content initialization
2. BEV-query spatial biasing

For the implementation details, see:

- [SAM3_BEVFORMER_13D_BEVPRIOR.md](./SAM3_BEVFORMER_13D_BEVPRIOR.md)

For environment setup, see:

- [ENVIRONMENT_SETUP.md](./ENVIRONMENT_SETUP.md)

For supplementary fusion notes and presentation figures, see:

- [Supplementary Documents](./docs/README.md)


## What Is Included

This codebase contains:

- the modified BEVFormer head and utilities
- SAM3 proposal loading and geometry construction
- experiment configs for `6D`, `13D`, `13D + BEV prior`, and mask-aware variants
- training launchers for the current best branch

The main files for the current method are:

- `projects/mmdet3d_plugin/datasets/pipelines/loading.py`
- `projects/mmdet3d_plugin/models/utils/sam_query_encoder.py`
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py`
- `projects/configs/bevformer/bevformer-tiny-sam3-query-13d-bevprior.py`
- `projects/configs/bevformer/bevformer-tiny-sam3-query-13d-bevprior-full.py`
- `tools/run_sam_bevprior_best.sh`
- `tools/run_sam_bevprior_full.sh`


## Method Summary

The method turns each SAM3 2D proposal into a 13D token:

```text
[xc, yc, w, h, score, class_id, cam_id,
 ray_dir_x, ray_dir_y, ray_dir_z,
 cam_pos_x, cam_pos_y, cam_pos_z]
```

These proposal tokens are used in two places:

### 1. SAM query initialization

`SAMProposalQueryEncoder` converts proposal geometry, category, camera identity, and ray features into learned query deltas. The first `K` BEVFormer object queries are initialized with these deltas.

### 2. BEV prior

`SAMBEVPriorEncoder` converts proposal rays into a soft BEV prior. This prior is mapped into BEV-query embeddings and added to the BEV query grid before the transformer runs.


## Expected External Assets

To reproduce training on a new machine, the recipient still needs:

1. nuScenes dataset
2. generated temporal info files
3. SAM3 detection JSON for the desired split
4. an initialization checkpoint for BEVFormer

Typical examples:

- dataset root: `data/nuscenes/`
- train ann: `data/nuscenes/nuscenes_infos_temporal_train.pkl`
- val ann: `data/nuscenes/nuscenes_infos_temporal_val.pkl`
- SAM3 JSON: `data/sam3/full/sam3_2d_detections_with_masks.json`
- init checkpoint: `work_dirs/bevformer_tiny_nuscenes/epoch_24.pth`


## Quick Start

### 1. Install environment

Follow:

- [ENVIRONMENT_SETUP.md](./ENVIRONMENT_SETUP.md)

### 2. Prepare data

Create a symlink or place the nuScenes data under:

```bash
data/nuscenes/
```

Make sure these files exist:

```text
data/nuscenes/nuscenes_infos_temporal_train.pkl
data/nuscenes/nuscenes_infos_temporal_val.pkl
```

### 3. Prepare SAM3 detections

For full-data training, provide a JSON file that stores SAM3 detections keyed by `sample_token` and camera name.

Default expected path:

```text
data/sam3/full/sam3_2d_detections_with_masks.json
```

You can override it with the environment variable:

```bash
export SAM3_RESULTS=/path/to/your/sam3_2d_detections_with_masks.json
```

### 4. Prepare initialization checkpoint

Set:

```bash
export LOAD_FROM=/path/to/your/bevformer_checkpoint.pth
```

If unset, the helper script defaults to:

```text
work_dirs/bevformer_tiny_nuscenes/epoch_24.pth
```


## Full Training Command

The cleanest way to train the current best method on full data is:

```bash
bash tools/run_sam_bevprior_full.sh work_dirs/sam3_query_bevprior_full 24
```

This launcher:

- activates the `bevformer` conda environment
- uses the full-data config
- loads the SAM3-guided `13D + BEV prior` method
- sets:
  - `sam_bev_prior_scale=0.05`
  - `sam_bev_prior_radius=2.0`

Useful overrides:

```bash
export SAM3_RESULTS=/path/to/full_sam3.json
export LOAD_FROM=/path/to/init_checkpoint.pth
export DATA_ROOT=/path/to/nuscenes
export TRAIN_ANN=/path/to/nuscenes_infos_temporal_train.pkl
export VAL_ANN=/path/to/nuscenes_infos_temporal_val.pkl
export TEST_ANN=/path/to/nuscenes_infos_temporal_val.pkl
```

Then run:

```bash
bash tools/run_sam_bevprior_full.sh work_dirs/sam3_query_bevprior_full 24
```

To run in background:

```bash
nohup bash tools/run_sam_bevprior_full.sh work_dirs/sam3_query_bevprior_full 24 \
  > work_dirs/sam3_query_bevprior_full.nohup.log 2>&1 &
```

To monitor:

```bash
tail -f work_dirs/sam3_query_bevprior_full.nohup.log
```


## Mini / Local Sanity Training

If you want a shorter run first, the local launcher used during development is:

```bash
bash tools/run_sam_bevprior_best.sh work_dirs/sam3_query_bevprior_best_train 12
```

That script is more tied to the original local setup. For external use, prefer `tools/run_sam_bevprior_full.sh`.


## Notes For External Users

- This package does not include the full dataset.
- This package does not include large training outputs by default.
- If you want exact reproduction, use the same dependency versions and CUDA stack from [ENVIRONMENT_SETUP.md](./ENVIRONMENT_SETUP.md).
- If you want to distribute checkpoints as well, send them separately from the code archive.


## Citation

If you use this branch in a paper or report, please also cite the original BEVFormer work and the corresponding SAM/SAM3 work you build on.
