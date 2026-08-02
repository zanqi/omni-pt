#!/bin/bash

#SBATCH --job-name=babble_data
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zanqil@uw.edu

#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=90G
#SBATCH --gpus=1
#SBATCH --time=01-0:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt
#SBATCH --output=logs/babble_data_%j.log
#SBATCH --export=all

# Thin wrapper -- see babble_data_qwen25.slurm. This file only pins the Qwen3
# resource request; track, dataset id and row counts come in as babble_data.sh
# env vars via --export=all.
#
#   TREE=1 N_EXTRA_ANS=1000 sbatch babble_data_qwen3.slurm
#   DS_QWEN3=me/my-ds sbatch babble_data_qwen3.slurm

exec ./babble_data.sh qwen3 "$@"
