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

SOURCE=experiments/cam-benchmarks.py
CKPT=./logs/classification/cifar10/train.scnn.randaug/2/saved_model


# EXPERIMENT=cam
# EXPERIMENT_TAGS="['cifar10', 'scnn', 'momentum', 'randaug', 'cam']"
# LOGS=./logs/explaining/cifar10/scnn.randaug

# docker-compose exec $SERVICE                                  \
#   python -X pycache_prefix=$PYTHONPYCACHEPREFIX $SOURCE with  \
#   config/explaining/base.yml                                  \
#   config/explaining/datasets/cifar10.yml                      \
#   config/training/scratch-training.yml                        \
#   config/augmentation/randaug.yml                             \
#   config/models/scnn.yml                                      \
#   config/classification/optimizers/momentum_nesterov.yml      \
#   config/logging/wandb.cam-benchmark.yml                      \
#   evaluation.config.pooling=scnn3.avg_pool                    \
#   evaluation.config.method=cam                                \
#   setup.paths.ckpt=$CKPT                                      \
#   setup.paths.wandb_dir=$LOGS                                 \
#   setup.wandb.name=$EXPERIMENT                                \
#   setup.wandb.tags="$EXPERIMENT_TAGS"                         \
#   -F $LOGS


EXPERIMENT=grad-cam++
EXPERIMENT_TAGS="['cifar10', 'scnn', 'momentum', 'randaug', 'grad-cam++']"
LOGS=./logs/explaining/cifar10/scnn.randaug

docker-compose exec $SERVICE                                  \
  python -X pycache_prefix=$PYTHONPYCACHEPREFIX $SOURCE with  \
  config/explaining/base.yml                                  \
  config/explaining/datasets/cifar10.yml                      \
  config/training/scratch-training.yml                        \
  config/augmentation/randaug.yml                             \
  config/models/scnn.yml                                      \
  config/classification/optimizers/momentum_nesterov.yml      \
  config/logging/wandb.cam-benchmark.yml                      \
  evaluation.config.pooling=scnn3.avg_pool                    \
  evaluation.config.method=gradcampp                          \
  setup.paths.ckpt=$CKPT                                      \
  setup.paths.wandb_dir=$LOGS                                 \
  setup.wandb.name=$EXPERIMENT                                \
  setup.wandb.tags="$EXPERIMENT_TAGS"                         \
  -F $LOGS


# EXPERIMENT=score-cam
# EXPERIMENT_TAGS="['cifar10', 'scnn', 'momentum', 'randaug', 'score-cam']"
# LOGS=./logs/explaining/cifar10/scnn.randaug

# docker-compose exec $SERVICE                                  \
#   python -X pycache_prefix=$PYTHONPYCACHEPREFIX $SOURCE with  \
#   config/explaining/base.yml                                  \
#   config/explaining/datasets/cifar10.yml                      \
#   config/training/scratch-training.yml                        \
#   config/augmentation/randaug.yml                             \
#   config/models/scnn.yml                                      \
#   config/classification/optimizers/momentum_nesterov.yml      \
#   config/logging/wandb.cam-benchmark.yml                      \
#   evaluation.config.pooling=scnn3.avg_pool                    \
#   evaluation.config.method=scorecam                           \
#   setup.paths.ckpt=$CKPT                                      \
#   setup.paths.wandb_dir=$LOGS                                 \
#   setup.wandb.name=$EXPERIMENT                                \
#   setup.wandb.tags="$EXPERIMENT_TAGS"                         \
#   -F $LOGS
