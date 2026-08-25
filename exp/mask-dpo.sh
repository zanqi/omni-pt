#!/bin/bash

#SBATCH --job-name=mask_dpo
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zanqil@uw.edu

#SBATCH --account=sciencehub
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=16:00:00

#SBATCH --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt
#SBATCH --output=logs/mask_dpo_%j.log
#SBATCH --export=all

# Both DPO stages in one job (steps/mask.html steps 9-10): sample + judge the
# preference pairs from the SFT checkpoint, then train on them. They share a
# GPU and the second is worthless without the first, so splitting them across
# two jobs only adds a queue wait.
#
#   sbatch exp/mask-dpo.sh
#   SFT_ADAPTER=checkpoints/my-sft RUN_NAME=my-dpo sbatch exp/mask-dpo.sh
#   K=16 sbatch exp/mask-dpo.sh         # more samples per row
#   SKIP_PREFS=1 sbatch exp/mask-dpo.sh # reuse an existing prefs file

source ~/.bashrc
set -eo pipefail

DS_ID="${DS_ID:-keylazy/slurp-mask-v1}"
SFT_ADAPTER="${SFT_ADAPTER:-checkpoints/Qwen2.5-Omni-3B-mask-sft}"
RUN_NAME="${RUN_NAME:-Qwen2.5-Omni-3B-mask-dpo}"
K="${K:-8}"
PREFS="${PREFS:-results/mask_prefs_$(basename "$SFT_ADAPTER").jsonl}"

conda activate qwen25omni

# fail-fast: the pairs are ranked by the judge box, and a run that cannot reach
# it produces no pairs at all -- an hour of sampling in
JUDGE_HOST=$(cat /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge_host.txt)
JUDGE_URL="http://${JUDGE_HOST}:8000/v1"
if ! curl -sf --max-time 10 "${JUDGE_URL}/models" > /dev/null; then
    echo "vLLM judge not reachable at ${JUDGE_URL} -- start it: sbatch /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge.slurm" >&2
    exit 1
fi
# the box gets re-served with different models; a stale name is a 404 on every
# judge call, which reads as "every sample scored 0"
JUDGE_MODEL=$(curl -sf --max-time 10 "${JUDGE_URL}/models" \
    | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
echo "judge OK at ${JUDGE_URL} serving ${JUDGE_MODEL}"

if [[ -z "${SKIP_PREFS:-}" ]]; then
    echo "=== sampling ${K}x from ${SFT_ADAPTER} -> ${PREFS} ==="
    python -u mask_dpo_data.py \
        --ds-id "$DS_ID" \
        --adapter-path "$SFT_ADAPTER" \
        --out "$PREFS" \
        -k "$K" \
        --judge-base-url "$JUDGE_URL" \
        --judge-model "$JUDGE_MODEL"
fi

echo "=== DPO ${RUN_NAME} on ${PREFS} $* ==="
python -u dpo_qwen.py \
    --ds-id "$DS_ID" \
    --prefs "$PREFS" \
    --sft-adapter "$SFT_ADAPTER" \
    --run-name "$RUN_NAME" \
    "$@"
echo "=== ${RUN_NAME} trained ==="
