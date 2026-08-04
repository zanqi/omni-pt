#!/bin/bash

#SBATCH --job-name=qwen25_sft
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zanqil@uw.edu

#SBATCH --account=cse
# Klone rejects multi-partition submissions (it silently keeps the first), so
# pick one here and override per-submission with `sbatch -p gpu-a100 ...`.
# --account=cse is remapped to the matching gpu-<part>-cse association, so the
# override needs no -A. cse's a100 association is usually CPU-capped; l40s
# almost always has room.
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=01-0:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt
#SBATCH --output=logs/qwen25_sft_%j.log
#SBATCH --export=all

# Thin wrapper: track and sweep shape are sft.sh env vars, carried in from the
# submitting shell by --export=all. conda activate and LD_LIBRARY_PATH live in
# sft.sh, so they stay in one place.
#
#   MULTS= TREE=1 sbatch sft_qwen25.slurm    # ONE adapter on the full split
#   TREE=1 sbatch sft_qwen25.slurm           # the 1x-4x answer-row sweep
#   HR=1 sbatch sft_qwen25.slurm
#
# Both the --chdir and the script name were stale before this: they pointed at
# asr-calibration/qwen_omni/sft and ran slurp_sft_qwen25.py, which does not
# exist in this repo.

exec ./sft.sh qwen25 "$@"
