#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=256
#SBATCH -p sequana_gpu_shared
#SBATCH -J lerdl-ss-train.enb0.randaug.regularization
#SBATCH -o /scratch/lerdl/lucas.david/logs/cifar10.train.enb0.randaug.regularization.%j.out
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
# Train EfficientNetB0 to perform multiclass classification task over Cifar10.
# Four regularization strategies are tested:
#   - None (baseline)
#   - L1L2 (weak)
#   - Orthogonal (strong)
#   - Kernel Usage (ours, strong)
#
# Make sure the run-specific parameters are added:
#
#   EXPERIMENT=enb0-randaug.regularization
#   EXPERIMENT_TAGS="['cifar10', 'enb0', 'momentum']"
#   LOGS=./logs/classification/cifar10/train.enb0.randaug.regularization
#
# If resuming a failed experiment, remember to add:
#
#   setup.wandb.resume=True

echo "[cifar/train.enb0.randaug.regularization.sh] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana

cd $SCRATCH/salient-segmentation
source config/sdumont/.env

SOURCE=experiments/train_and_finetune.py
LOGS=$LOGS_DIR/classification/cifar10/regularization
mkdir -p $LOGS

EXPERIMENT=enb0-randaug
EXPERIMENT_TAGS="['cifar10', 'efficientnet', 'momentum', 'randaug']"
CUDA_VISIBLE_DEVICES=0 python3.9 $SOURCE with             \
  config/classification/train_and_finetune.yml            \
  config/training/preinitialized-training.yml             \
  config/models/enb0.yml                                  \
  config/classification/datasets/cifar10.yml              \
  config/augmentation/randaug.yml                         \
  config/classification/optimizers/momentum_nesterov.yml  \
  config/logging/wandb.train.yml                          \
  setup.paths.ckpt=$LOGS/backup                           \
  setup.paths.wandb_dir=$LOGS                             \
  setup.wandb.name=$EXPERIMENT                            \
  setup.wandb.tags="$EXPERIMENT_TAGS"                     \
  -F $LOGS                                                \
  &> $LOGS/$EXPERIMENT.log                                &


EXPERIMENT=enb0-randaug-dropout
EXPERIMENT_TAGS="['cifar10', 'efficientnet', 'momentum', 'randaug', 'dropout']"
CUDA_VISIBLE_DEVICES=1 python3.9 $SOURCE with             \
  config/classification/train_and_finetune.yml            \
  config/training/preinitialized-training.yml             \
  config/models/enb0.yml                                  \
  config/classification/datasets/cifar10.yml              \
  config/augmentation/randaug.yml                         \
  config/classification/optimizers/momentum_nesterov.yml  \
  config/classification/regularizers/dropout.yml          \
  config/logging/wandb.train.yml                          \
  setup.paths.ckpt=$LOGS/backup                           \
  setup.paths.wandb_dir=$LOGS                             \
  setup.wandb.name=$EXPERIMENT                            \
  setup.wandb.tags="$EXPERIMENT_TAGS"                     \
  -F $LOGS                                                \
  &> $LOGS/$EXPERIMENT.log                                &


EXPERIMENT=enb0-randaug-l1l2
EXPERIMENT_TAGS="['cifar10', 'efficientnet', 'momentum', 'randaug', 'l1l2']"
CUDA_VISIBLE_DEVICES=2 python3.9 $SOURCE with             \
  config/classification/train_and_finetune.yml            \
  config/training/preinitialized-training.yml             \
  config/models/enb0.yml                                  \
  config/classification/datasets/cifar10.yml              \
  config/augmentation/randaug.yml                         \
  config/classification/optimizers/momentum_nesterov.yml  \
  config/classification/regularizers/l1l2.yml             \
  config/logging/wandb.train.yml                          \
  setup.paths.ckpt=$LOGS/backup                           \
  setup.paths.wandb_dir=$LOGS                             \
  setup.wandb.name=$EXPERIMENT                            \
  setup.wandb.tags="$EXPERIMENT_TAGS"                     \
  -F $LOGS                                                \
  &> $LOGS/$EXPERIMENT.log                                &


EXPERIMENT=enb0-randaug-ortho
EXPERIMENT_TAGS="['cifar10', 'efficientnet', 'momentum', 'randaug', 'ortho']"
CUDA_VISIBLE_DEVICES=3 python3.9 $SOURCE with             \
  config/classification/train_and_finetune.yml            \
  config/training/preinitialized-training.yml             \
  config/models/enb0.yml                                  \
  config/classification/datasets/cifar10.yml              \
  config/augmentation/randaug.yml                         \
  config/classification/optimizers/momentum_nesterov.yml  \
  config/classification/regularizers/orthogonal.yml       \
  config/logging/wandb.train.yml                          \
  setup.paths.ckpt=$LOGS/backup                           \
  setup.paths.wandb_dir=$LOGS                             \
  setup.wandb.name=$EXPERIMENT                            \
  setup.wandb.tags="$EXPERIMENT_TAGS"                     \
  -F $LOGS                                                \
  &> $LOGS/$EXPERIMENT.log                                &


EXPERIMENT=enb0-randaug-kernel-usage
EXPERIMENT_TAGS="['cifar10', 'efficientnet', 'momentum', 'randaug', 'kernel-usage']"
CUDA_VISIBLE_DEVICES=3 python3.9 $SOURCE with             \
  config/classification/train_and_finetune.yml            \
  config/training/preinitialized-training.yml             \
  config/models/enb0.yml                                  \
  config/classification/datasets/cifar10.yml              \
  config/augmentation/randaug.yml                         \
  config/classification/optimizers/momentum_nesterov.yml  \
  config/classification/regularizers/kernel-usage.yml     \
  config/logging/wandb.train.yml                          \
  setup.paths.ckpt=$LOGS/backup                           \
  setup.paths.wandb_dir=$LOGS                             \
  setup.wandb.name=$EXPERIMENT                            \
  setup.wandb.tags="$EXPERIMENT_TAGS"                     \
  -F $LOGS                                                \
  &> $LOGS/$EXPERIMENT.log                                &

wait


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
