#!/usr/bin/env bash
set -eo pipefail

ROOT=/122090720/bevformer/BEVFormer
CONFIG=${CONFIG:-projects/configs/bevformer/bevformer-tiny-sam3-query-13d-bevprior.py}
WORKDIR=${1:-work_dirs/sam3_query_bevprior_best_train}
EPOCHS=${2:-12}
LOAD_FROM=${LOAD_FROM:-work_dirs/bevformer_tiny_nuscenes_mini/epoch_24.pth}

source /122090720/miniconda3/etc/profile.d/conda.sh
conda activate bevformer

cd "$ROOT"
export PYTHONPATH="$ROOT"
unset CC CXX CUDAHOSTCXX

python tools/train.py \
  "$CONFIG" \
  --work-dir "$WORKDIR" \
  --cfg-options \
    total_epochs="$EPOCHS" \
    runner.max_epochs="$EPOCHS" \
    data.workers_per_gpu=0 \
    data.train.data_root=/122090720/datasets/nuscenes/ \
    data.train.ann_file=/122090720/datasets/nuscenes/nuscenes_infos_temporal_train.pkl \
    data.val.data_root=/122090720/datasets/nuscenes/ \
    data.val.ann_file=/122090720/datasets/nuscenes/nuscenes_infos_temporal_val.pkl \
    data.test.data_root=/122090720/datasets/nuscenes/ \
    data.test.ann_file=/122090720/datasets/nuscenes/nuscenes_infos_temporal_val.pkl \
    load_from="$LOAD_FROM" \
    model.pts_bbox_head.sam_bev_prior_scale=0.05 \
    model.pts_bbox_head.sam_bev_prior_radius=2.0
