#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH -p sequana_gpu_shared
#SBATCH -J ss_setup
#SBATCH -o /scratch/lerdl/lucas.david/logs/setup.%j.out
#SBATCH --time=03:00:00

nodeset -e $SLURM_JOB_NODELIST

module load sequana/current
module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana

SRC_DIR=$SCRATCH/salient-segmentation

cd $SRC_DIR
source config/sdumont/.env

# python3.9 -m pip install -U pip
pip3.9 install -r requirements.txt
