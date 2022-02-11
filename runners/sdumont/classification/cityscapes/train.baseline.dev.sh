#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH -p nvidia_dev
#SBATCH -J pbn_train_cityscapes_dev
#SBATCH --exclusive
#SBATCH -o /scratch/lerdl/lucas.david/logs/cityscapes/baseline/train-cityscapes-dev-%j.out



echo "[train.cityscapes.dev.sh] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST
module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

CODE_DIR=$SCRATCH/experiments
BUILD_DIR=$SCRATCH/experiments/build
CONFIG_DIR=$SCRATCH/experiments/config/classification.cityscapes.dev.yml
LOGS_DIR=$SCRATCH/logs/cityscapes/baseline/

cd $CODE_DIR

python3.9 -X pycache_prefix=$BUILD_DIR src/baseline.py with $CONFIG_DIR -F $LOGS_DIR
