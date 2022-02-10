#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -p sequana_gpu_shared
#SBATCH -J lerdl_voc07_rn101_randaug_all
#SBATCH -o /scratch/lerdl/lucas.david/logs/voc07/rn101-randaug/%j.out
#SBATCH --time=24:00:00

echo "[train.voc07.randaug.sh] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana

SRC_DIR=$SCRATCH/salient-segmentation
BUILD_DIR=$SRC_DIR/build
CONFIG_DIR=$SRC_DIR/config/classification/voc07/rn101.randaug.finetune.yml
LOGS_DIR=$SCRATCH/logs/voc07/rn101-randaug
DATA_DIR=$SCRATCH/datasets

cd $SRC_DIR

# CUDA_VISIBLE_DEVICES=0 python3.9 -X pycache_prefix=$BUILD_DIR src/baseline.py \
#   with $CONFIG_DIR                                                            \
#   model.head.dropout_rate=0                                                   \
#   dataset.load.data_dir=$DATA_DIR                                             \
#   training.perform=False                                                      \
#   setup.paths.data=$DATA_DIR                                                  \
#   setup.paths.ckpt=$LOGS_DIR/baseline/backup                                  \
#   -F $LOGS_DIR/baseline                                                       \
#   > $LOGS_DIR/baseline.log 2>&1                                               &

CUDA_VISIBLE_DEVICES=0 python3.9 -X pycache_prefix=$BUILD_DIR src/baseline.py \
  with $CONFIG_DIR                                                            \
  model.head.dropout_rate=0                                                   \
  training.finetune.unfreeze.freeze_bn=False                                  \
  dataset.load.data_dir=$DATA_DIR                                             \
  training.perform=False                                                      \
  setup.paths.data=$DATA_DIR                                                  \
  setup.paths.ckpt=$LOGS_DIR/baseline-fbn/backup                              \
  -F $LOGS_DIR/baseline-fbn                                                   \
  > $LOGS_DIR/baseline-fbn.log 2>&1                                           &

CUDA_VISIBLE_DEVICES=1 python3.9 -X pycache_prefix=$BUILD_DIR src/baseline.py \
  with $CONFIG_DIR                                                            \
  model.head.dropout_rate=0                                                   \
  model.head.kernel_regularizer=l2                                            \
  dataset.load.data_dir=$DATA_DIR                                             \
  training.perform=False                                                      \
  setup.paths.data=$DATA_DIR                                                  \
  setup.paths.ckpt=$LOGS_DIR/l2/backup                                        \
  -F $LOGS_DIR/l2 > $LOGS_DIR/l2.log 2>&1                                     &

CUDA_VISIBLE_DEVICES=2 python3.9 -X pycache_prefix=$BUILD_DIR src/baseline.py \
  with $CONFIG_DIR                                                            \
  model.head.dropout_rate=0                                                   \
  model.head.kernel_initializer=orthogonal                                    \
  model.head.kernel_regularizer=orthogonal                                    \
  dataset.load.data_dir=$DATA_DIR                                             \
  training.perform=False                                                      \
  setup.paths.data=$DATA_DIR                                                  \
  setup.paths.ckpt=$LOGS_DIR/ortho/backup                                     \
  -F $LOGS_DIR/ortho > $LOGS_DIR/ortho.log 2>&1                               &

CUDA_VISIBLE_DEVICES=3 python3.9 -X pycache_prefix=$BUILD_DIR src/baseline.py \
  with $SRC_DIR/config/classification/voc07/rn101.randaug.kur.yml             \
  model.head.dropout_rate=0                                                   \
  model.head.layer_class=kernel_usage                                         \
  dataset.load.data_dir=$DATA_DIR                                             \
  training.perform=False                                                      \
  setup.paths.data=$DATA_DIR                                                  \
  setup.paths.ckpt=$LOGS_DIR/kur/backup                                       \
  -F $LOGS_DIR/kur > $LOGS_DIR/kur.log 2>&1                                   &

wait
