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
# Convert Checkpoint to Keras Model.
#

source config/docker/.env

SOURCE=experiments/infra/convert_checkpoint_to_keras_model.py

docker-compose exec $SERVICE                              \
  python -X pycache_prefix=$PYTHONPYCACHEPREFIX $SOURCE with  \
  config/infra/convert-checkpoint-to-keras-model.yml \
  config/classification/datasets/coco17.yml               \
  config/models/rn101.yml                                 \
  config/classification/regularizers/kernel-usage.yml     \
  config/classification/optimizers/momentum_nesterov.yml  \
  config/training/preinitialized-training.yml             \
  setup.paths.ckpt=/tf/logs/models/coco-2017/coco17-kernel-usage  \
  setup.paths.export=/tf/logs/models/coco-2017/rn101-kur          \
  config/environment/precision-mixed-float16.yml


#   All Mixins Available:
#
#   config/classification/datasets/cifar10.yml              \
#   config/classification/datasets/cityscapes.yml           \
#   config/classification/datasets/coco17.yml               \
#   config/classification/datasets/voc07.yml                \
#   config/classification/datasets/voc12.yml                \
#
#   config/classification/optimizers/momentum_nesterov.yml  \
#
#   config/training/preinitialized-training.yml             \
#   config/training/train-head-and-finetune.yml             \
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
