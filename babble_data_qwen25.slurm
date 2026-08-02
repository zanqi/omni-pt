#!/bin/bash

#SBATCH --job-name=babble_data
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zanqil@uw.edu

#SBATCH --account=intelligentsystems # cse
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=188G
#SBATCH --gpus=1
#SBATCH --time=01-0:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt
#SBATCH --output=logs/babble_data_%j.log
#SBATCH --export=all

# Thin wrapper: everything that varies -- track, dataset id, row counts -- is a
# babble_data.sh env var, and --export=all carries it in from the submitting
# shell. This file only pins the Qwen2.5 resource request; the judge precheck
# and the conda activate live in babble_data.sh, so they stay in one place.
#
#   TREE=1 N_EXTRA_ANS=1000 sbatch babble_data_qwen25.slurm
#   HR=1 sbatch babble_data_qwen25.slurm
#   DS_QWEN25=me/my-ds sbatch babble_data_qwen25.slurm
#   sbatch babble_data_qwen25.slurm            # default two-pass track

exec ./babble_data.sh qwen25 "$@"
