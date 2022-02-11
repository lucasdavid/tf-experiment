#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -p sequana_gpu_shared
#SBATCH -J ss_setup
#SBATCH -o /scratch/lerdl/lucas.david/logs/ss-%j.out
#SBATCH --time=03:00:00


nodeset -e $SLURM_JOB_NODELIST

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana

SRC_DIR=$SCRATCH/salient-segmentation
BUILD_DIR=$SRC_DIR/build
CONFIG_DIR=$SRC_DIR/config/classification.cityscapes.yml
LOGS_DIR=$SCRATCH/logs/cityscapes/baseline/

cd $SRC_DIR
source ../config/sdumont/.env

pip3.9 install -r requirements.txt
# python3.9 -X pycache_prefix=$BUILD_DIR src/baseline.py with $CONFIG_DIR -F $LOGS_DIR
