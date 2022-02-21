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
# Train ResNet101 to perform multilabel classification task over COCO 2017.

source config/local/.env
export PYTHONPATH="${PYTHONPATH}::./src"

EXPERIMENT=rn101-dropout
EXPERIMENT_TAGS="['coco17', 'rn101', 'momentum', 'randaug', 'dropout']"

SOURCE=experiments/train_and_finetune.py
LOGS=./logs/classification/coco17/train.rn101.dropout


python -X pycache_prefix=$PYTHONPYCACHEPREFIX $SOURCE print_config with  \
  config/classification/train_and_finetune.yml            \
  config/classification/datasets/coco17.yml               \
  config/classification/optimizers/momentum_nesterov.yml  \
  config/models/rn101.yml                                 \
  config/classification/regularizers/dropout.yml          \
  config/augmentation/randaug.yml                         \
  config/logging/wandb.train.yml                          \
  dataset.prepare.shuffle.buffer_size=5000                \
  setup.paths.ckpt=$LOGS/backup                           \
  setup.paths.wandb_dir=$LOGS                             \
  setup.wandb.name=$EXPERIMENT                            \
  setup.wandb.tags="$EXPERIMENT_TAGS"                     \
  -F $LOGS
