#!/bin/bash

#SBATCH --job-name=qwen25_sft
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zanqil@uw.edu

#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=01-0:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/gscratch/sciencehub/zanqil/asr-calibration/qwen_omni/sft
#SBATCH --export=all
source ~/.bashrc
conda activate qwen25omni
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python slurp_babble_data.py

# sbatch /gscratch/sciencehub/zanqil/asr-calibration/qwen_omni/sft/slurp_babble_data.slurm
