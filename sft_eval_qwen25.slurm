#!/bin/bash

#SBATCH --job-name=qwen25_sft_eval
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zanqil@uw.edu

#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=01-0:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt
#SBATCH --output=logs/qwen25_sft_eval_%j.log
#SBATCH --export=all

# Train ONE adapter on the train split exactly as built, then judge the base
# model and that adapter on the same test split -- in one job, so the eval
# starts the moment training finishes.
#
#   TREE=1 sbatch sft_eval_qwen25.slurm    # decision-table track
#   HR=1 sbatch sft_eval_qwen25.slurm      # Heard:/Reply: track
#   sbatch sft_eval_qwen25.slurm           # default two-pass track
#   MULTS="1 2" TREE=1 sbatch sft_eval_qwen25.slurm   # sweep those mixes instead
#
# Results land in results/bab_results_<name>_<tag>.jsonl, one for the base
# model and one for the adapter.

set -eo pipefail

# Single run is this script's default. It lives here rather than on the submit
# line because an empty MULTS= is exactly the kind of thing --export can drop,
# and silently falling back to the 4-run sweep would be an expensive surprise.
export MULTS="${MULTS-}"

./sft.sh qwen25
./eval.sh qwen25
