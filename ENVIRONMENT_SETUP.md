# Environment Setup

This document describes a practical environment setup for training the SAM3-guided BEVFormer branch on nuScenes.


## 1. System Dependencies

Install the common runtime libraries first:

```bash
apt-get update
apt-get install -y libgl1 libglib2.0-0
```

If your machine uses a custom CUDA stack, make sure the NVIDIA driver and CUDA runtime are compatible with your PyTorch installation.


## 2. Create the Conda Environment

```bash
conda create -n bevformer python=3.10 -y
conda activate bevformer
```


## 3. Install PyTorch

The development environment used in this branch used:

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

If you need a different CUDA build, replace the index URL and versions accordingly, but keep the rest of the stack aligned.


## 4. Install Detectron2

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```


## 5. Install MMCV From This Repository

This repo contains a local MMCV tree with custom ops support.

```bash
cd mmcv
MMCV_WITH_OPS=1 pip install . --no-cache-dir -v
cd ..
```


## 6. Install MMDetection3D / BEVFormer Package

From the repository root:

```bash
pip install -e .
```


## 7. Runtime Environment Variables

Before training or evaluation, use:

```bash
export PYTHONPATH=$(pwd)
unset CC CXX CUDAHOSTCXX
```

If you use the provided launcher scripts, they already do this for you.


## 8. Data Layout

Recommended layout:

```text
BEVFormer/
  data/
    nuscenes/
      nuscenes_infos_temporal_train.pkl
      nuscenes_infos_temporal_val.pkl
      samples/
      sweeps/
      maps/
      v1.0-trainval/
```

You can either place the dataset there directly or create a symlink:

```bash
mkdir -p data
ln -s /path/to/nuscenes data/nuscenes
```


## 9. SAM3 Detection File

For full training, prepare a SAM3 result JSON and place it at:

```text
data/sam3/full/sam3_2d_detections_with_masks.json
```

or point to it via:

```bash
export SAM3_RESULTS=/path/to/sam3_2d_detections_with_masks.json
```

The loader expects a JSON object containing:

- `detections`
- keyed by `sample_token`
- then keyed by camera name such as `CAM_FRONT`

Each detection entry should contain at least:

- `bbox`
- `label`
- `score`

Optional mask-aware branches can additionally use:

- `mask_rle`


## 10. Initialization Checkpoint

Training in this repo is expected to start from a BEVFormer checkpoint rather than from scratch.

Set:

```bash
export LOAD_FROM=/path/to/bevformer_init_checkpoint.pth
```

If you use the provided full-training launcher and do not set `LOAD_FROM`, it defaults to:

```text
work_dirs/bevformer_tiny_nuscenes/epoch_24.pth
```


## 11. Smoke Checks

Before a long run, these are useful sanity checks:

```bash
python -m py_compile projects/mmdet3d_plugin/datasets/pipelines/loading.py
python -m py_compile projects/mmdet3d_plugin/models/utils/sam_query_encoder.py
python -m py_compile projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py
```

If you want to test the query-only branch tooling:

```bash
python tools/sam_queryonly_mask_smoke.py
```


## 12. Common Failure Points

### Missing GUI/OpenCV runtime libraries

Symptom:

- OpenCV import errors
- `libGL.so` missing

Fix:

```bash
apt-get install -y libgl1 libglib2.0-0
```

### Conda deactivate hook errors

If your shell environment injects strict `set -u` behavior, some conda deactivate hooks may fail on unset variables. The provided launchers avoid the problematic shell settings.

### Wrong absolute paths

This repo originally lived on a machine with absolute paths under `/122090720/...`. For external use, prefer the portable launcher scripts and environment variables instead of copying those hardcoded paths.


## 13. Recommended Launch Path

After setup, the recommended full-data training entrypoint is:

```bash
bash tools/run_sam_bevprior_full.sh work_dirs/sam3_query_bevprior_full 24
```

For the method details, see:

- [SAM3_BEVFORMER_13D_BEVPRIOR.md](./SAM3_BEVFORMER_13D_BEVPRIOR.md)
