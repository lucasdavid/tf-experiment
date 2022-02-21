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
# Train a simple CNN to perform multiclass classification task over Cifar10.
#

source config/docker/.env

SOURCE=experiments/train_and_finetune.py

EXPERIMENT=scnn-noaug
EXPERIMENT_TAGS="['cifar10', 'scnn', 'momentum']"
LOGS=./logs/classification/cifar10/train.scnn.noaug

docker-compose exec $SERVICE                                  \
  python -X pycache_prefix=$PYTHONPYCACHEPREFIX $SOURCE with  \
  config/classification/train_and_finetune.yml            \
  config/classification/datasets/cifar10.yml              \
  config/training/scratch-training.yml                    \
  config/augmentation/none.yml                            \
  config/models/scnn.yml                                  \
  config/classification/optimizers/momentum_nesterov.yml  \
  config/logging/wandb.train.yml                          \
  config/environment/wandb.dev.yml                        \
  setup.wandb.name=$EXPERIMENT                            \
  setup.wandb.tags="$EXPERIMENT_TAGS"                     \
  setup.paths.wandb_dir=$LOGS                             \
  setup.paths.ckpt=$LOGS/backup                           \
  -F $LOGS
  
# EXPERIMENT=scnn-randaug
# EXPERIMENT_TAGS="['cifar10', 'scnn', 'momentum', 'randaug']"
# LOGS=./logs/classification/cifar10/train.scnn.randaug

# docker-compose exec $SERVICE                                  \
#   python -X pycache_prefix=$PYTHONPYCACHEPREFIX $SOURCE with  \
#   config/classification/train_and_finetune.yml            \
#   config/classification/datasets/cifar10.yml              \
#   config/training/scratch-training.yml                    \
#   config/augmentation/randaug.yml                         \
#   config/models/scnn.yml                                  \
#   config/classification/optimizers/momentum_nesterov.yml  \
#   config/logging/wandb.train.yml                          \
#   config/environment/wandb.dev.yml                        \
#   setup.wandb.name=$EXPERIMENT                            \
#   setup.wandb.tags="$EXPERIMENT_TAGS"                     \
#   setup.paths.wandb_dir=$LOGS                             \
#   setup.paths.ckpt=$LOGS/backup                           \
#   -F $LOGS
