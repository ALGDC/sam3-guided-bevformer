#!/usr/bin/env bash
set -eo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=${CONFIG:-projects/configs/bevformer/bevformer-tiny-sam3-query-13d-bevprior-full.py}
WORKDIR=${1:-work_dirs/sam3_query_bevprior_full}
EPOCHS=${2:-24}

CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-bevformer}

DATA_ROOT=${DATA_ROOT:-data/nuscenes/}
TRAIN_ANN=${TRAIN_ANN:-${DATA_ROOT%/}/nuscenes_infos_temporal_train.pkl}
VAL_ANN=${VAL_ANN:-${DATA_ROOT%/}/nuscenes_infos_temporal_val.pkl}
TEST_ANN=${TEST_ANN:-${DATA_ROOT%/}/nuscenes_infos_temporal_val.pkl}
SAM3_RESULTS=${SAM3_RESULTS:-data/sam3/full/sam3_2d_detections_with_masks.json}
LOAD_FROM=${LOAD_FROM:-work_dirs/bevformer_tiny_nuscenes/epoch_24.pth}

if [ ! -f "$CONDA_SH" ]; then
  echo "Missing conda init script: $CONDA_SH" >&2
  exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$ROOT"
export PYTHONPATH="$ROOT"
export SAM3_RESULTS
unset CC CXX CUDAHOSTCXX

python tools/train.py \
  "$CONFIG" \
  --work-dir "$WORKDIR" \
  --cfg-options \
    total_epochs="$EPOCHS" \
    runner.max_epochs="$EPOCHS" \
    data.train.data_root="$DATA_ROOT" \
    data.train.ann_file="$TRAIN_ANN" \
    data.val.data_root="$DATA_ROOT" \
    data.val.ann_file="$VAL_ANN" \
    data.test.data_root="$DATA_ROOT" \
    data.test.ann_file="$TEST_ANN" \
    load_from="$LOAD_FROM"
