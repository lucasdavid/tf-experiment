#!/bin/bash

# Copyright 2021 Lucas Oliveira David
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Benchmark CAM Techniques using ResNet50 over VOC 2007.
#

source config/docker/.env

EXPERIMENT=cam
EXPERIMENT_TAGS="['voc07', 'rn50', 'momentum', 'randaug', 'cam']"

SOURCE=experiments/cam-benchmarks.py
CKPT=./logs/classification/voc07/rn50.randaug/train/3/saved_model
LOGS=./logs/explaining/voc07/rn50.randaug


docker-compose exec $SERVICE                                  \
  python -X pycache_prefix=$PYTHONPYCACHEPREFIX $SOURCE with  \
  config/explaining/base.yml                                  \
  config/explaining/datasets/voc07.yml                        \
  config/augmentation/randaug.yml                             \
  config/classification/optimizers/momentum_nesterov.yml      \
  evaluation.config.pooling=resnet50v2.avg_pool \
  dataset.prepare.take=1   \
  setup.paths.ckpt=$CKPT                                      \
  -F $LOGS

# dataset.prepare.batch_size=16   \

#  config/logging/wandb.cam-benchmark.yml                  \
#
# setup.paths.wandb_dir=$LOGS                             \
# setup.wandb.name=$EXPERIMENT                            \
# setup.wandb.tags="$EXPERIMENT_TAGS"                     \
