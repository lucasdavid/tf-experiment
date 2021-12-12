#!/bin/bash

echo "[train.voc07.sh] started running at $(date +'%Y-%m-%d %H:%M:%S')."

export SCRATCH=/tf
export CODE_DIR=$SCRATCH/src
export BUILD_DIR=$SCRATCH/build
export DATA_DIR=$SCRATCH/data/painter-by-numbers
export LOGS_DIR=$SCRATCH/logs
export WEIGHTS_DIR=$SCRATCH/logs/models


python -X pycache_prefix=$BUILD_DIR src/baseline.py \
  with ./config/classification.voc07.yml            \
  -F ./logs/voc07/rn50-baseline/
