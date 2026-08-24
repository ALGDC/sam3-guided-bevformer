#!/usr/bin/env bash
set -eo pipefail

ROOT=/122090720/bevformer/BEVFormer
CONFIG=projects/configs/bevformer/bevformer-tiny-sam3-query-15d-queryonly.py
WORKDIR=${1:-work_dirs/sam3_query_queryonly_mask}

source /122090720/miniconda3/etc/profile.d/conda.sh
conda activate bevformer

cd "$ROOT"
export PYTHONPATH="$ROOT"
unset CC CXX CUDAHOSTCXX

python tools/train.py \
  "$CONFIG" \
  --work-dir "$WORKDIR" \
  --cfg-options \
    total_epochs=1 \
    runner.max_epochs=1 \
    data.workers_per_gpu=0 \
    data.train.data_root=/122090720/datasets/nuscenes/ \
    data.train.ann_file=/122090720/datasets/nuscenes/nuscenes_infos_temporal_train.pkl \
    data.val.data_root=/122090720/datasets/nuscenes/ \
    data.val.ann_file=/122090720/datasets/nuscenes/nuscenes_infos_temporal_val.pkl \
    data.test.data_root=/122090720/datasets/nuscenes/ \
    data.test.ann_file=/122090720/datasets/nuscenes/nuscenes_infos_temporal_val.pkl \
    load_from=work_dirs/bevformer_tiny_nuscenes_mini/epoch_24.pth
