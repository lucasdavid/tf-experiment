#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=256
#SBATCH -p sequana_gpu_shared
#SBATCH -J lerdl-ss-coco17-train.rn101.randaug.preinitialized-and-finetune
#SBATCH -o /scratch/lerdl/lucas.david/logs/coco17.train.rn101.randaug.preinitialized-and-finetune.%j.out
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
# Train ResNet50 to perform multilabel classification task over COCO 2017.
#
# Four regularization strategies are tested:
#   - Dropout (baseline)
#   - Kernel Usage (ours, strong)
#
# Make sure the run-specific parameters are added:
#
#   EXPERIMENT=rn101-randaug
#   EXPERIMENT_TAGS="['coco17', 'rn101', 'momentum']"
#   LOGS=./logs/classification/coco17/train.rn101.all
#
# If resuming a failed experiment, remember to add:
#
#   setup.wandb.resume=True

echo "[coco17/train.rn101.all.sh] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana

cd $SCRATCH/salient-segmentation

set -o allexport
source ./config/sdumont/.env
set +o allexport

SOURCE=experiments/train_and_finetune.py

# =====================================================================
# Preinitialized training

EXPERIMENT=rn101-dropout
EXPERIMENT_TAGS="['coco17', 'rn101', 'momentum', 'randaug', 'dropout']"
LOGS=$LOGS_DIR/classification/coco17/regularization
mkdir -p $LOGS
CUDA_VISIBLE_DEVICES=0 python3.9 $SOURCE with             \
  config/classification/train_and_finetune.yml                   \
  config/classification/datasets/coco17.yml               \
  config/augmentation/randaug.yml                         \
  config/classification/optimizers/momentum_nesterov.yml  \
  config/models/rn101.yml                                 \
  config/classification/regularizers/dropout.yml          \
  config/training/preinitialized-training.yml             \
  config/environment/precision-mixed-float16.yml          \
  config/environment/sdumont.yml                          \
  config/logging/wandb.train.yml                          \
  setup.paths.ckpt=$LOGS/backup/$EXPERIMENT                           \
  setup.paths.wandb_dir=$LOGS                             \
  setup.wandb.name=$EXPERIMENT                            \
  setup.wandb.tags="$EXPERIMENT_TAGS"                     \
  -F $LOGS                                                \
  &> $LOGS/$EXPERIMENT.log                                &
echo "Job [$EXPERIMENT] stacked. Logs will be placed at $LOGS/$EXPERIMENT.log"


EXPERIMENT=rn101-kernel-usage
EXPERIMENT_TAGS="['coco17', 'rn101', 'momentum', 'randaug', 'kernel-usage']"
LOGS=$LOGS_DIR/classification/coco17/regularization
mkdir -p $LOGS
CUDA_VISIBLE_DEVICES=1 python3.9 $SOURCE with             \
  config/classification/train_and_finetune.yml                   \
  config/classification/datasets/coco17.yml               \
  config/augmentation/randaug.yml                         \
  config/classification/optimizers/momentum_nesterov.yml  \
  config/models/rn101.yml                                 \
  config/classification/regularizers/kernel-usage.yml     \
  model.head.config.alpha=133                                         \
  config/training/preinitialized-training.yml             \
  config/environment/precision-mixed-float16.yml          \
  config/environment/sdumont.yml                          \
  config/logging/wandb.train.yml                          \
  setup.paths.ckpt=$LOGS/backup/$EXPERIMENT                           \
  setup.paths.wandb_dir=$LOGS                             \
  setup.wandb.name=$EXPERIMENT                            \
  setup.wandb.tags="$EXPERIMENT_TAGS"                     \
  -F $LOGS                                                \
  &> $LOGS/$EXPERIMENT.log                                &
echo "Job [$EXPERIMENT] stacked. Logs will be placed at $LOGS/$EXPERIMENT.log"

# =====================================================================
## Head Training and Finetuning with Frozen Batch Norm

EXPERIMENT=rn101-finetune-dropout
EXPERIMENT_TAGS="['coco17', 'rn101', 'finetune', 'momentum', 'randaug', 'dropout']"
LOGS=$LOGS_DIR/classification/coco17/regularization
mkdir -p $LOGS
CUDA_VISIBLE_DEVICES=2 python3.9 $SOURCE with             \
  config/classification/train_and_finetune.yml                   \
  config/classification/datasets/coco17.yml               \
  config/augmentation/randaug.yml                         \
  config/classification/optimizers/momentum_nesterov.yml  \
  config/models/rn101.yml                                 \
  config/classification/regularizers/dropout.yml          \
  config/training/train-head-and-finetune.yml             \
  config/environment/precision-mixed-float16.yml          \
  config/environment/sdumont.yml                          \
  config/logging/wandb.train.yml                          \
  setup.paths.ckpt=$LOGS/backup/$EXPERIMENT                           \
  setup.paths.wandb_dir=$LOGS                             \
  setup.wandb.name=$EXPERIMENT                            \
  setup.wandb.tags="$EXPERIMENT_TAGS"                     \
  -F $LOGS                                                \
  &> $LOGS/$EXPERIMENT.log                                &
echo "Job [$EXPERIMENT] stacked. Logs will be placed at $LOGS/$EXPERIMENT.log"


EXPERIMENT=rn101-finetune-kernel-usage
EXPERIMENT_TAGS="['coco17', 'rn101', 'finetune', 'momentum', 'randaug', 'kernel-usage']"
LOGS=$LOGS_DIR/classification/coco17/regularization
mkdir -p $LOGS
CUDA_VISIBLE_DEVICES=3 python3.9 $SOURCE with             \
  config/classification/train_and_finetune.yml                   \
  config/classification/datasets/coco17.yml               \
  config/augmentation/randaug.yml                         \
  config/classification/optimizers/momentum_nesterov.yml  \
  config/models/rn101.yml                                 \
  config/classification/regularizers/kernel-usage.yml     \
  model.head.config.alpha=133                                         \
  config/training/train-head-and-finetune.yml             \
  config/environment/precision-mixed-float16.yml          \
  config/environment/sdumont.yml                          \
  config/logging/wandb.train.yml                          \
  setup.paths.ckpt=$LOGS/backup/$EXPERIMENT                           \
  setup.paths.wandb_dir=$LOGS                             \
  setup.wandb.name=$EXPERIMENT                            \
  setup.wandb.tags="$EXPERIMENT_TAGS"                     \
  -F $LOGS                                                \
  &> $LOGS/$EXPERIMENT.log                                &
echo "Job [$EXPERIMENT] stacked. Logs will be placed at $LOGS/$EXPERIMENT.log"


wait


#   All Mixins Available:
#
#   config/classification/datasets/cifar10.yml              \
#   config/classification/datasets/cifar100.yml             \
#   config/classification/datasets/cityscapes.yml           \
#   config/classification/datasets/coco17.yml               \
#   config/classification/datasets/voc07.yml                \
#   config/classification/datasets/voc12.yml                \
#
#   config/augmentation/none.yml                            \
#   config/augmentation/simple.yml                          \
#   config/augmentation/randaug.yml                         \
#
#   config/classification/optimizers/momentum_nesterov.yml  \
#
#   config/models/enb0.yml                                  \
#   config/models/enb6.yml                                  \
#   config/models/rn50.yml                                  \
#   config/models/rn101.yml                                 \
#   config/models/rn152.yml                                 \
#
#   config/training/preinitialized-training.yml             \
#   config/training/scratch-training.yml                    \
#   config/training/train-head-and-finetune.yml             \
#
#   config/classification/regularizers/dropout.yml          \
#   config/classification/regularizers/kernel-usage.yml     \
#   config/classification/regularizers/orthogonal.yml       \
#   config/classification/regularizers/l1l2.yml             \
#
#   config/environment/precision-mixed-float16.yml          \
#   config/environment/dev.yml                              \
#   config/environment/sdumont.yml                          \
#   config/logging/wandb.train.yml                          \
