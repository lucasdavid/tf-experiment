#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -p sequana_gpu_shared
#SBATCH -J lerdl_voc07_rn101_randaug_all
#SBATCH -o /scratch/lerdl/lucas.david/logs/mixed/rn101-randaug/%j.out
#SBATCH --time=24:00:00

echo "[train.voc07.randaug.sh] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana

SRC_DIR=$SCRATCH/salient-segmentation
BUILD_DIR=$SRC_DIR/build
LOGS_DIR=$SCRATCH/logs/mixed/rn101-randaug
DATA_DIR=$SCRATCH/datasets

cd $SRC_DIR

CUDA_VISIBLE_DEVICES=0 python3.9 -X pycache_prefix=$BUILD_DIR src/baseline.py \
  with $SRC_DIR/config/classification/voc07/rn101.randaug.yml                 \
  training.finetune.unfreeze.freeze_bn=False                                  \
  model.head.dropout_rate=0                                                   \
  dataset.load.data_dir=$DATA_DIR                                             \
  setup.paths.data=$DATA_DIR                                                  \
  setup.paths.ckpt=$LOGS_DIR/voc07-baseline/backup                            \
  -F $LOGS_DIR/voc07-baseline                                                 \
  &> $LOGS_DIR/voc07-baseline.log                                             &

CUDA_VISIBLE_DEVICES=1 python3.9 -X pycache_prefix=$BUILD_DIR src/baseline.py \
  with $SRC_DIR/config/classification/voc07/rn101.randaug.kur.yml             \
  training.finetune.unfreeze.freeze_bn=False                                  \
  model.head.dropout_rate=0                                                   \
  dataset.load.data_dir=$DATA_DIR                                             \
  setup.paths.data=$DATA_DIR                                                  \
  setup.paths.ckpt=$LOGS_DIR/voc07-kur/backup                                 \
  -F $LOGS_DIR/voc07-kur                                                      \
  &> $LOGS_DIR/voc07-kur.log                                                  &

CUDA_VISIBLE_DEVICES=2 python3.9 -X pycache_prefix=$BUILD_DIR src/baseline.py \
  with $SRC_DIR/config/classification/coco17/rn101.randaug.yml                \
  training.finetune.unfreeze.freeze_bn=False                                  \
  model.head.dropout_rate=0                                                   \
  dataset.load.data_dir=$DATA_DIR                                             \
  setup.paths.data=$DATA_DIR                                                  \
  setup.paths.ckpt=$LOGS_DIR/coco17-baseline/backup                           \
  -F $LOGS_DIR/coco17-baseline                                                \
  &> $LOGS_DIR/coco17-baseline.log                                            &

CUDA_VISIBLE_DEVICES=3 python3.9 -X pycache_prefix=$BUILD_DIR src/baseline.py \
  with $SRC_DIR/config/classification/coco17/rn101.randaug.kur.yml            \
  training.finetune.unfreeze.freeze_bn=False                                  \
  model.head.dropout_rate=0                                                   \
  dataset.load.data_dir=$DATA_DIR                                             \
  setup.paths.data=$DATA_DIR                                                  \
  setup.paths.ckpt=$LOGS_DIR/coco17-kur/backup                                \
  -F $LOGS_DIR/coco17-kur                                                     \
  &> $LOGS_DIR/coco17-kur.log                                                 &

wait
