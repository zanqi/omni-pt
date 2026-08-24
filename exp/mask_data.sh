#!/bin/bash

#SBATCH --job-name=mask_data
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zanqil@uw.edu

#SBATCH --account=sciencehub
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=12:00:00

#SBATCH --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt
#SBATCH --output=logs/mask_data_%j.log
#SBATCH --export=all

# Build the mask-track dataset (steps/mask.html): one answer row and one
# repair row per utterance, over the same background.
#
#   sbatch exp/mask_data.sh                          # 1500 train / 80 test -> DS_ID
#   N_TRAIN=200 N_TEST=20 sbatch exp/mask_data.sh    # a shorter run
#   DS_ID=me/my-ds sbatch exp/mask_data.sh           # override the pushed repo
#   sbatch exp/mask_data.sh --no-push --asr-model openai/whisper-tiny.en
#
# Anything after the script name is passed straight to mask_data.py, so every
# flag it has (--mask-pad, --clean-bg-prob, --silence-hard, --exclude-ds, ...)
# works without a knob here.
#
# One A40 is plenty: the builder loads no omni model, only the 0.6B aligner and
# the gate's whisper. The wall clock is per-utterance GPU work plus two target
# calls per train utterance to the vLLM box.

source ~/.bashrc
set -eo pipefail

DS_ID="${DS_ID:-keylazy/slurp-mask-v1}"

# unset means mask_data.py's own defaults
COUNTS=""
[[ -n "${N_TRAIN:-}" ]]     && COUNTS="$COUNTS --n-train $N_TRAIN"
[[ -n "${N_TEST:-}" ]]      && COUNTS="$COUNTS --n-test $N_TEST"

# transformers is a dev build here (the forced aligner is not in a release yet)
# and openai-whisper is installed alongside it
conda activate qwen3omni

# fail-fast: the train split's targets come from the vLLM box, and a build that
# cannot reach it produces test rows and nothing else -- an hour in.
# mask_data.py reads the host file itself and takes the *served* model name off
# /v1/models, so a re-served box needs no edit here.
TARGET_HOST=$(cat /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge_host.txt)
TARGET_URL="http://${TARGET_HOST}:8000/v1"
if ! curl -sf --max-time 10 "${TARGET_URL}/models" > /dev/null; then
    echo "vLLM box not reachable at ${TARGET_URL} -- start it: sbatch /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge.slurm" >&2
    exit 1
fi
echo "target model box OK at ${TARGET_URL} serving $(curl -sf --max-time 10 "${TARGET_URL}/models" \
    | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"

echo "=== building ${DS_ID}${COUNTS} $* ==="
python -u mask_data.py --ds-id "$DS_ID" $COUNTS "$@"
echo "=== ${DS_ID} built ==="
