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
#   EXPERIMENT_TAGS="['voc07', 'rn50', 'momentum']"
#   LOGS=./logs/classification/voc07/train.rn50.noaug
#
# If resuming a failed experiment, remember to add:
#
#   setup.wandb.resume=True                                             \

source config/docker/.env

EXPERIMENT=rn50-randaug
EXPERIMENT_TAGS="['voc07', 'rn50', 'momentum', 'randaug']"

SOURCE=src/train_and_finetune.py
LOGS=./logs/classification/voc07/train.rn50.randaug


docker-compose exec $SERVICE python $SOURCE                           \
  with config/runs/classification/train_and_finetune.yml              \
  config/runs/classification/mixins/datasets/voc07.yml                \
  config/runs/mixins/augmentation/randaug.yml                         \
  config/runs/classification/mixins/optimizers/momentum_nesterov.yml  \
  config/runs/mixins/logging/wandb.yml                                \
  setup.paths.ckpt=$LOGS/backup                                       \
  setup.paths.wandb_dir=$LOGS_DIR/wandb                               \
  setup.wandb.name=$EXPERIMENT                                        \
  setup.wandb.tags="$EXPERIMENT_TAGS"                                 \
  -F $LOGS


#   All Mixins Available:
#
#   config/runs/classification/mixins/datasets/cifar10.yml              \
#   config/runs/classification/mixins/datasets/cityscapes.yml           \
#   config/runs/classification/mixins/datasets/coco17.yml               \
#   config/runs/classification/mixins/datasets/voc07.yml                \
#   config/runs/classification/mixins/datasets/voc12.yml                \
#
#   config/runs/classification/mixins/optimizers/momentum_nesterov.yml  \
#
#   config/runs/classification/mixins/regularizers/dropout.yml          \
#   config/runs/classification/mixins/regularizers/kernel-usage.yml     \
#   config/runs/classification/mixins/regularizers/orthogonal.yml       \
#   config/runs/classification/mixins/regularizers/l1l2.yml             \
#
#   config/runs/mixins/augmentation/none.yml                            \
#   config/runs/mixins/augmentation/simple.yml                          \
#   config/runs/mixins/augmentation/randaug.yml                         \
#
#   config/runs/mixins/environment/precision-mixed-float16.yml          \
#   config/runs/mixins/environment/dev.yml                              \
#   config/runs/mixins/logging/wandb.yml                                \
#
#   config/runs/mixins/models/enb0.yml                                  \
#   config/runs/mixins/models/enb6.yml                                  \
#   config/runs/mixins/models/rn50.yml                                  \
#   config/runs/mixins/models/rn101.yml                                 \
