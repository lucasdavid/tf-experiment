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
# Train ResNet50 to perform multilabel classification task over VOC 2007.
#

source config/docker/.env

EXPERIMENT=rn50-randaug
EXPERIMENT_TAGS="['voc07', 'rn50', 'momentum', 'randaug']"

SOURCE=experiments/train_and_finetune.py
LOGS=./logs/classification/voc07/train.rn50.randaug


docker-compose exec $SERVICE                                  \
  python -X pycache_prefix=$PYTHONPYCACHEPREFIX $SOURCE with  \
  config/classification/train_and_finetune.yml            \
  config/classification/datasets/voc07.yml                \
  config/augmentation/randaug.yml                         \
  config/classification/optimizers/momentum_nesterov.yml  \
  config/logging/wandb.train.yml                          \
  setup.paths.ckpt=$LOGS/backup                           \
  setup.paths.wandb_dir=$LOGS                             \
  setup.wandb.name=$EXPERIMENT                            \
  setup.wandb.tags="$EXPERIMENT_TAGS"                     \
  -F $LOGS
