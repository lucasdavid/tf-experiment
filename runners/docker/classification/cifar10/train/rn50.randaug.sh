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
# Train ResNet50 to perform multiclass classification task over Cifar10.
#
# You can check the configuration mixure before running by adding
# `print_config` after $SOURCE below:
#
# ```bash
# docker-compose exec $SERVICE python $SOURCE print_config ...
# ```
#
# Make sure the run-specific parameters are added:
#
#   EXPERIMENT=rn50-noaug
#   EXPERIMENT_TAGS="['cifar10', 'rn50', 'momentum']"
#   LOGS=./logs/classification/cifar10/train.rn50.noaug
#
# If resuming a failed experiment, remember to add:
#
#   setup.wandb.resume=True

source config/docker/.env

EXPERIMENT=rn50-randaug
EXPERIMENT_TAGS="['cifar10', 'rn50', 'momentum', 'randaug']"

SOURCE=experiments/train_and_finetune.py
LOGS=./logs/classification/cifar10/train.rn50.randaug


docker-compose exec $SERVICE                              \
  python -X pycache_prefix=$PYTHONPYCACHEPREFIX $SOURCE with  \
  config/classification/train_and_finetune.yml            \
  config/classification/datasets/cifar10.yml              \
  config/augmentation/randaug.yml                         \
  config/classification/optimizers/momentum_nesterov.yml  \
  config/logging/wandb.train.yml                          \
  setup.paths.ckpt=$LOGS/backup                           \
  setup.paths.wandb_dir=$LOGS                             \
  setup.wandb.name=$EXPERIMENT                            \
  setup.wandb.tags="$EXPERIMENT_TAGS"                     \
  -F $LOGS


#   All Mixins Available:
#
#   config/classification/datasets/cifar10.yml              \
#   config/classification/datasets/cityscapes.yml           \
#   config/classification/datasets/coco17.yml               \
#   config/classification/datasets/voc07.yml                \
#   config/classification/datasets/voc12.yml                \
#
#   config/training/preinitialized-training.yml             \
#   config/training/train-head-and-finetune.yml             \
#
#   config/classification/optimizers/momentum_nesterov.yml  \
#
#   config/classification/regularizers/dropout.yml          \
#   config/classification/regularizers/kernel-usage.yml     \
#   config/classification/regularizers/orthogonal.yml       \
#   config/classification/regularizers/l1l2.yml             \
#
#   config/augmentation/none.yml                            \
#   config/augmentation/simple.yml                          \
#   config/augmentation/randaug.yml                         \
#
#   config/environment/precision-mixed-float16.yml          \
#   config/environment/dev.yml                              \
#   config/logging/wandb.train.yml                          \
#
#   config/models/enb0.yml                                  \
#   config/models/enb6.yml                                  \
#   config/models/rn50.yml                                  \
#   config/models/rn101.yml                                 \
#   config/models/rn152.yml                                 \
