#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -p sequana_gpu_shared
#SBATCH -J lerdl-ss-train.rn50.randaug.regularization
#SBATCH -o /scratch/lerdl/lucas.david/logs/cifar10.train.rn50.randaug.regularization.%j.out
#SBATCH --time=96:00:00

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
# Four regularization strategies are tested:
#   - None (baseline)
#   - L1L2 (weak)
#   - Orthogonal (strong)
#   - Kernel Usage (massive)
#
# Make sure the run-specific parameters are added:
#
#   EXPERIMENT=rn50-randaug.regularization
#   EXPERIMENT_TAGS="['cifar10', 'rn50', 'momentum']"
#   LOGS=./logs/classification/cifar10/train.rn50.randaug.regularization
#
# If resuming a failed experiment, remember to add:
#
#   setup.wandb.resume=True

echo "[cifar/train.rn50.randaug.regularization.sh] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana

cd $SCRATCH/salient-segmentation
source config/sdumont/.env

SOURCE=src/train_and_finetune.py
LOGS=$LOGS_DIR/classification/cifar10/regularization
mkdir -p $LOGS

EXPERIMENT=rn50-randaug
EXPERIMENT_TAGS="['cifar10', 'rn50', 'momentum', 'randaug']"
CUDA_VISIBLE_DEVICES=0 python3.9 $SOURCE with                         \
  with config/runs/classification/train_and_finetune.yml              \
  config/runs/mixins/training/preinitialized-training.yml             \
  config/runs/mixins/models/rn50.yml                                  \
  config/runs/classification/mixins/datasets/cifar10.yml              \
  config/runs/mixins/augmentation/randaug.yml                         \
  config/runs/classification/mixins/optimizers/momentum_nesterov.yml  \
  config/runs/mixins/logging/wandb.yml                                \
  setup.paths.ckpt=$LOGS/backup                                       \
  setup.paths.wandb_dir=$LOGS_DIR/wandb                               \
  setup.wandb.name=$EXPERIMENT                                        \
  setup.wandb.tags="$EXPERIMENT_TAGS"                                 \
  -F $LOGS                                                            \
  &> $LOGS/$EXPERIMENT.log                                            &


EXPERIMENT=rn50-randaug-dropout
EXPERIMENT_TAGS="['cifar10', 'rn50', 'momentum', 'randaug', 'dropout']"
CUDA_VISIBLE_DEVICES=1 python3.9 $SOURCE with                         \
  with config/runs/classification/train_and_finetune.yml              \
  config/runs/mixins/training/preinitialized-training.yml             \
  config/runs/mixins/models/rn50.yml                                  \
  config/runs/classification/mixins/datasets/cifar10.yml              \
  config/runs/mixins/augmentation/randaug.yml                         \
  config/runs/classification/mixins/optimizers/momentum_nesterov.yml  \
  config/runs/classification/mixins/regularizers/dropout.yml          \
  config/runs/mixins/logging/wandb.yml                                \
  setup.paths.ckpt=$LOGS/backup                                       \
  setup.paths.wandb_dir=$LOGS_DIR/wandb                               \
  setup.wandb.name=$EXPERIMENT                                        \
  setup.wandb.tags="$EXPERIMENT_TAGS"                                 \
  -F $LOGS                                                            \
  &> $LOGS/$EXPERIMENT.log                                            &


EXPERIMENT=rn50-randaug-l1l2
EXPERIMENT_TAGS="['cifar10', 'rn50', 'momentum', 'randaug', 'l1l2']"
CUDA_VISIBLE_DEVICES=2 python3.9 $SOURCE with                         \
  with config/runs/classification/train_and_finetune.yml              \
  config/runs/mixins/training/preinitialized-training.yml             \
  config/runs/mixins/models/rn50.yml                                  \
  config/runs/classification/mixins/datasets/cifar10.yml              \
  config/runs/mixins/augmentation/randaug.yml                         \
  config/runs/classification/mixins/optimizers/momentum_nesterov.yml  \
  config/runs/classification/mixins/regularizers/l1l2.yml             \
  config/runs/mixins/logging/wandb.yml                                \
  setup.paths.ckpt=$LOGS/backup                                       \
  setup.paths.wandb_dir=$LOGS_DIR/wandb                               \
  setup.wandb.name=$EXPERIMENT                                        \
  setup.wandb.tags="$EXPERIMENT_TAGS"                                 \
  -F $LOGS                                                            \
  &> $LOGS/$EXPERIMENT.log                                            &


EXPERIMENT=rn50-randaug-ortho
EXPERIMENT_TAGS="['cifar10', 'rn50', 'momentum', 'randaug', 'ortho']"
CUDA_VISIBLE_DEVICES=3 python3.9 $SOURCE with                         \
  with config/runs/classification/train_and_finetune.yml              \
  config/runs/mixins/training/preinitialized-training.yml             \
  config/runs/mixins/models/rn50.yml                                  \
  config/runs/classification/mixins/datasets/cifar10.yml              \
  config/runs/mixins/augmentation/randaug.yml                         \
  config/runs/classification/mixins/optimizers/momentum_nesterov.yml  \
  config/runs/classification/mixins/regularizers/orthogonal.yml       \
  config/runs/mixins/logging/wandb.yml                                \
  setup.paths.ckpt=$LOGS/backup                                       \
  setup.paths.wandb_dir=$LOGS_DIR/wandb                               \
  setup.wandb.name=$EXPERIMENT                                        \
  setup.wandb.tags="$EXPERIMENT_TAGS"                                 \
  -F $LOGS                                                            \
  &> $LOGS/$EXPERIMENT.log                                            &


EXPERIMENT=rn50-randaug-kernel-usage
EXPERIMENT_TAGS="['cifar10', 'rn50', 'momentum', 'randaug', 'kernel-usage']"
CUDA_VISIBLE_DEVICES=3 python3.9 $SOURCE with                         \
  with config/runs/classification/train_and_finetune.yml              \
  config/runs/mixins/training/preinitialized-training.yml             \
  config/runs/mixins/models/rn50.yml                                  \
  config/runs/classification/mixins/datasets/cifar10.yml              \
  config/runs/mixins/augmentation/randaug.yml                         \
  config/runs/classification/mixins/optimizers/momentum_nesterov.yml  \
  config/runs/classification/mixins/regularizers/kernel-usage.yml     \
  config/runs/mixins/logging/wandb.yml                                \
  setup.paths.ckpt=$LOGS/backup                                       \
  setup.paths.wandb_dir=$LOGS_DIR/wandb                               \
  setup.wandb.name=$EXPERIMENT                                        \
  setup.wandb.tags="$EXPERIMENT_TAGS"                                 \
  -F $LOGS                                                            \
  &> $LOGS/$EXPERIMENT.log                                            &

wait


#   All Mixins Available:
#
#   config/runs/classification/mixins/datasets/cifar10.yml              \
#   config/runs/classification/mixins/datasets/cityscapes.yml           \
#   config/runs/classification/mixins/datasets/coco17.yml               \
#   config/runs/classification/mixins/datasets/voc07.yml                \
#   config/runs/classification/mixins/datasets/voc12.yml                \
#
#   config/runs/mixins/training/preinitialized-training.yml             \
#   config/runs/mixins/training/train-head-and-finetune.yml             \
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
#   config/runs/mixins/models/rn152.yml                                 \
