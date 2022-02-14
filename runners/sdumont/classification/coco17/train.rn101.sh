#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
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

SOURCE=src/train_and_finetune.py

# =====================================================================
# Preinitialized training

EXPERIMENT=rn101-dropout
EXPERIMENT_TAGS="['coco17', 'rn101', 'momentum', 'randaug', 'dropout']"
LOGS=$LOGS_DIR/classification/coco17/regularization
mkdir -p $LOGS
CUDA_VISIBLE_DEVICES=0 python3.9 $SOURCE with                         \
  config/runs/classification/train_and_finetune.yml                   \
  config/runs/classification/mixins/datasets/coco17.yml               \
  config/runs/mixins/augmentation/randaug.yml                         \
  config/runs/classification/mixins/optimizers/momentum_nesterov.yml  \
  config/runs/mixins/models/rn101.yml                                 \
  config/runs/classification/mixins/regularizers/dropout.yml          \
  config/runs/mixins/training/preinitialized-training.yml             \
  config/runs/mixins/environment/precision-mixed-float16.yml          \
  config/runs/mixins/environment/sdumont.yml                          \
  config/runs/mixins/logging/wandb.yml                                \
  setup.paths.ckpt=$LOGS/backup/$EXPERIMENT                           \
  setup.paths.wandb_dir=$LOGS_DIR                                     \
  setup.wandb.name=$EXPERIMENT                                        \
  setup.wandb.tags="$EXPERIMENT_TAGS"                                 \
  setup.wandb.resume=True                                             \
  -F $LOGS                                                            \
  &> $LOGS/$EXPERIMENT.log                                            &
echo "Job [$EXPERIMENT] stacked. Logs will be placed at $LOGS/$EXPERIMENT.log"


EXPERIMENT=rn101-kernel-usage
EXPERIMENT_TAGS="['coco17', 'rn101', 'momentum', 'randaug', 'kernel-usage']"
LOGS=$LOGS_DIR/classification/coco17/regularization
mkdir -p $LOGS
CUDA_VISIBLE_DEVICES=1 python3.9 $SOURCE with                         \
  config/runs/classification/train_and_finetune.yml                   \
  config/runs/classification/mixins/datasets/coco17.yml               \
  config/runs/mixins/augmentation/randaug.yml                         \
  config/runs/classification/mixins/optimizers/momentum_nesterov.yml  \
  config/runs/mixins/models/rn101.yml                                 \
  config/runs/classification/mixins/regularizers/kernel-usage.yml     \
  model.head.config.alpha=133                                         \
  config/runs/mixins/training/preinitialized-training.yml             \
  config/runs/mixins/environment/precision-mixed-float16.yml          \
  config/runs/mixins/environment/sdumont.yml                          \
  config/runs/mixins/logging/wandb.yml                                \
  setup.paths.ckpt=$LOGS/backup/$EXPERIMENT                           \
  setup.paths.wandb_dir=$LOGS_DIR                                     \
  setup.wandb.name=$EXPERIMENT                                        \
  setup.wandb.tags="$EXPERIMENT_TAGS"                                 \
  setup.wandb.resume=True                                             \
  -F $LOGS                                                            \
  &> $LOGS/$EXPERIMENT.log                                            &
echo "Job [$EXPERIMENT] stacked. Logs will be placed at $LOGS/$EXPERIMENT.log"

# =====================================================================
## Head Training and Finetuning with Frozen Batch Norm

EXPERIMENT=rn101-finetune-dropout
EXPERIMENT_TAGS="['coco17', 'rn101', 'finetune', 'momentum', 'randaug', 'dropout']"
LOGS=$LOGS_DIR/classification/coco17/regularization
mkdir -p $LOGS
CUDA_VISIBLE_DEVICES=2 python3.9 $SOURCE with                         \
  config/runs/classification/train_and_finetune.yml                   \
  config/runs/classification/mixins/datasets/coco17.yml               \
  config/runs/mixins/augmentation/randaug.yml                         \
  config/runs/classification/mixins/optimizers/momentum_nesterov.yml  \
  config/runs/mixins/models/rn101.yml                                 \
  config/runs/classification/mixins/regularizers/dropout.yml          \
  config/runs/mixins/training/train-head-and-finetune.yml             \
  config/runs/mixins/environment/precision-mixed-float16.yml          \
  config/runs/mixins/environment/sdumont.yml                          \
  config/runs/mixins/logging/wandb.yml                                \
  setup.paths.ckpt=$LOGS/backup/$EXPERIMENT                           \
  setup.paths.wandb_dir=$LOGS_DIR                                     \
  setup.wandb.name=$EXPERIMENT                                        \
  setup.wandb.tags="$EXPERIMENT_TAGS"                                 \
  setup.wandb.resume=True                                             \
  -F $LOGS                                                            \
  &> $LOGS/$EXPERIMENT.log                                            &
echo "Job [$EXPERIMENT] stacked. Logs will be placed at $LOGS/$EXPERIMENT.log"


EXPERIMENT=rn101-finetune-kernel-usage
EXPERIMENT_TAGS="['coco17', 'rn101', 'finetune', 'momentum', 'randaug', 'kernel-usage']"
LOGS=$LOGS_DIR/classification/coco17/regularization
mkdir -p $LOGS
CUDA_VISIBLE_DEVICES=3 python3.9 $SOURCE with                         \
  config/runs/classification/train_and_finetune.yml                   \
  config/runs/classification/mixins/datasets/coco17.yml               \
  config/runs/mixins/augmentation/randaug.yml                         \
  config/runs/classification/mixins/optimizers/momentum_nesterov.yml  \
  config/runs/mixins/models/rn101.yml                                 \
  config/runs/classification/mixins/regularizers/kernel-usage.yml     \
  model.head.config.alpha=133                                         \
  config/runs/mixins/training/train-head-and-finetune.yml             \
  config/runs/mixins/environment/precision-mixed-float16.yml          \
  config/runs/mixins/environment/sdumont.yml                          \
  config/runs/mixins/logging/wandb.yml                                \
  setup.paths.ckpt=$LOGS/backup/$EXPERIMENT                           \
  setup.paths.wandb_dir=$LOGS_DIR                                     \
  setup.wandb.name=$EXPERIMENT                                        \
  setup.wandb.tags="$EXPERIMENT_TAGS"                                 \
  setup.wandb.resume=True                                             \
  -F $LOGS                                                            \
  &> $LOGS/$EXPERIMENT.log                                            &
echo "Job [$EXPERIMENT] stacked. Logs will be placed at $LOGS/$EXPERIMENT.log"


wait


#   All Mixins Available:
#
#   config/runs/classification/mixins/datasets/cifar10.yml              \
#   config/runs/classification/mixins/datasets/cifar100.yml             \
#   config/runs/classification/mixins/datasets/cityscapes.yml           \
#   config/runs/classification/mixins/datasets/coco17.yml               \
#   config/runs/classification/mixins/datasets/voc07.yml                \
#   config/runs/classification/mixins/datasets/voc12.yml                \
#
#   config/runs/mixins/augmentation/none.yml                            \
#   config/runs/mixins/augmentation/simple.yml                          \
#   config/runs/mixins/augmentation/randaug.yml                         \
#
#   config/runs/classification/mixins/optimizers/momentum_nesterov.yml  \
#
#   config/runs/mixins/models/enb0.yml                                  \
#   config/runs/mixins/models/enb6.yml                                  \
#   config/runs/mixins/models/rn50.yml                                  \
#   config/runs/mixins/models/rn101.yml                                 \
#   config/runs/mixins/models/rn152.yml                                 \
#
#   config/runs/mixins/training/preinitialized-training.yml             \
#   config/runs/mixins/training/scratch-training.yml                    \
#   config/runs/mixins/training/train-head-and-finetune.yml             \
#
#   config/runs/classification/mixins/regularizers/dropout.yml          \
#   config/runs/classification/mixins/regularizers/kernel-usage.yml     \
#   config/runs/classification/mixins/regularizers/orthogonal.yml       \
#   config/runs/classification/mixins/regularizers/l1l2.yml             \
#
#   config/runs/mixins/environment/precision-mixed-float16.yml          \
#   config/runs/mixins/environment/dev.yml                              \
#   config/runs/mixins/environment/sdumont.yml                          \
#   config/runs/mixins/logging/wandb.yml                                \