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
# Evaluate ResNet50 over the Cifar10 multiclass classification task.
#

source config/docker/.env

EXPERIMENT=rn50-noaug
EXPERIMENT_TAGS="['cifar10', 'rn50', 'momentum']"

SOURCE=experiments/evaluate.py
CKPT=./logs/classification/cifar10/rn50.noaug/train/2/saved_model
LOGS=./logs/classification/cifar10/rn50.noaug/evaluate


docker-compose exec $SERVICE                                  \
  python -X pycache_prefix=$PYTHONPYCACHEPREFIX $SOURCE with  \
  config/classification/evaluate.yml                      \
  config/classification/datasets/cifar10.yml              \
  config/classification/optimizers/momentum_nesterov.yml  \
  config/logging/wandb.evaluate.yml                       \
  config/environment/wandb.dev.yml                        \
  dataset.prepare.take=10                                 \
  setup.paths.export=$CKPT                                \
  setup.wandb.name=$EXPERIMENT                            \
  setup.wandb.tags="$EXPERIMENT_TAGS"                     \
  setup.paths.wandb_dir=$LOGS                             \
  setup.wandb.resume=True \
  -F $LOGS
  