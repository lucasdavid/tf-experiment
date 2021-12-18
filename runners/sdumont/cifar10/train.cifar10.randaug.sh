#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH -p nvidia_long
#SBATCH -J pbn_train_cifar10
#SBATCH --exclusive
#SBATCH -o /scratch/lerdl/lucas.david/logs/cifar10/baseline/train-cifar10-%j.out
#SBATCH --time=12:00:00


echo "[train.cifar10.sh] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST
module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

CODE_DIR=$SCRATCH/experiments
BUILD_DIR=$SCRATCH/experiments/build
CONFIG_DIR=$SCRATCH/experiments/config/classification/cifar10.randaug.yml
LOGS_DIR=$SCRATCH/logs/cifar10/baseline/

cd $CODE_DIR

python3.9 -X pycache_prefix=$BUILD_DIR src/baseline.py with $CONFIG_DIR -F $LOGS_DIR

# Regularized
python3.9 -X pycache_prefix=$BUILD_DIR src/baseline.py with $CONFIG_DIR model.head.kernel_regularizer=l2 model.head.dropout_rate=0 -F $LOGS_DIR
python3.9 -X pycache_prefix=$BUILD_DIR src/baseline.py with $CONFIG_DIR model.head.kernel_initializer=orthogonal model.head.kernel_regularizer=orthogonal model.head.dropout_rate=0 -F $LOGS_DIR
python3.9 -X pycache_prefix=$BUILD_DIR src/baseline.py with $CONFIG_DIR model.head.layer_class=kernel_usage model.head.dropout_rate=0 -F $LOGS_DIR
python3.9 -X pycache_prefix=$BUILD_DIR src/baseline.py with $CONFIG_DIR model.head.layer_class=kernel_usage model.head.dropout_rate=0 \
  training.optimizer.config.learning_rate=1.0          \
  training.finetune.optimizer.config.learning_rate=1.0 \
  -F $LOGS_DIR
