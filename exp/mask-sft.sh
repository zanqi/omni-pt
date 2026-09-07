#!/bin/bash

#SBATCH --job-name=mask_sft
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zanqil@uw.edu

#SBATCH --account=sciencehub
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=12:00:00

#SBATCH --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt
#SBATCH --output=logs/mask_sft_%j.log
#SBATCH --export=all

# LoRA SFT on the mask dataset (steps/mask.html step 8). No code change was
# needed -- sft_qwen.py already takes --repair-ds-id and filters by --train-kinds.
#
#   sbatch exp/mask-sft.sh
#   DS_ID=me/my-ds RUN_NAME=my-sft sbatch exp/mask-sft.sh
#   sbatch exp/mask-sft.sh --repair-epochs 2 --repair-lr 1e-4   # passthrough
#
# Adapter lands in checkpoints/$RUN_NAME and is pushed to keylazy/$RUN_NAME.

source ~/.bashrc
set -eo pipefail

DS_ID="${DS_ID:-keylazy/slurp-mask-v1}"
RUN_NAME="${RUN_NAME:-Qwen2.5-Omni-3B-mask-sft}"
# every kind the dataset carries. A v1 build has no repeat rows, so naming the
# kind here is harmless there and is what trains F on a v2 build.
TRAIN_KINDS="${TRAIN_KINDS:-answer,repair,repeat}"

conda activate qwen25omni

echo "=== SFT ${RUN_NAME} on ${DS_ID} $* ==="
python -u sft_qwen.py \
    --repair-ds-id "$DS_ID" \
    --repair-repo-name "$RUN_NAME" \
    --train-kinds "$TRAIN_KINDS" \
    "$@"
echo "=== ${RUN_NAME} trained ==="
