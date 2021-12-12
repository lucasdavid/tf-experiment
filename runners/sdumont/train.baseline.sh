#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH -p nvidia_long
#SBATCH -J pbn_train_voc07
#SBATCH --exclusive
#SBATCH -o /scratch/lerdl/lucas.david/experiments/logs/voc07/rn50-baseline/train-voc07-%j.out
#SBATCH --time=3-00:00:00


echo "[train.voc07.sh] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST
module load gcc/7.4 python/3.9.1 cudnn/8.2_cuda-11.1

CODE_DIR=$SCRATCH/experiments
BUILD_DIR=$SCRATCH/experiments/build

cd $CODE_DIR

python -X pycache_prefix=$BUILD_DIR src/baseline.py \
  with ./config/classification.voc07.yml            \
  -F ./logs/voc07/rn50-baseline/
