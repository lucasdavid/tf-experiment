#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH -p nvidia_long
#SBATCH -J pbn_train_cityscapes
#SBATCH --exclusive
#SBATCH -o /scratch/lerdl/lucas.david/logs/cityscapes/baseline/train-cityscapes-%j.out
#SBATCH --time=3-00:00:00


echo "[train.cityscapes.sh] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST
module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

CODE_DIR=$SCRATCH/experiments
BUILD_DIR=$SCRATCH/experiments/build
CONFIG_DIR=$SCRATCH/experiments/config/classification.cityscapes.yml
LOGS_DIR=$SCRATCH/logs/cityscapes/baseline/

cd $CODE_DIR

python3.9 -X pycache_prefix=$BUILD_DIR src/baseline.py with $CONFIG_DIR \
  dataset.load.splits="['train_extra','validation','train']" \
  -F $LOGS_DIR
