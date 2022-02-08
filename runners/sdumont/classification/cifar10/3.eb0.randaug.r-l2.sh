#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -p sequana_gpu_shared
#SBATCH -J ss_setup
#SBATCH -o /scratch/lerdl/lucas.david/logs/cifar10/eb0-randaug/%j.out
#SBATCH --time=3:00:00

echo "[train.cifar10.sh] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana

SRC_DIR=$SCRATCH/salient-segmentation
BUILD_DIR=$SRC_DIR/build
CONFIG_DIR=$SRC_DIR/config/classification/cifar10/eb0.randaug.yml
LOGS_DIR=$SCRATCH/logs/cifar10/eb0-randaug/

cd $SRC_DIR

python3.9 -X pycache_prefix=$BUILD_DIR src/baseline.py with $CONFIG_DIR \
  model.head.kernel_regularizer=l2 \
  model.head.dropout_rate=0 \
  setup.paths.ckpt=./logs/cifar10/eb0-randaug-l2/backup \
  -F $LOGS_DIR
