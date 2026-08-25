#!/bin/bash

#SBATCH --job-name=mask_eval
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zanqil@uw.edu

#SBATCH --account=sciencehub
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=12:00:00

#SBATCH --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt
#SBATCH --output=logs/mask_eval_%j.log
#SBATCH --export=all

# C / R / EAR at each stage of the mask track (steps/mask.html step 11), all on
# one test split with one judge so the three lines are directly comparable.
#
#   sbatch exp/mask.sh                       # base, SFT, DPO
#   STAGES="base sft" sbatch exp/mask.sh     # before DPO exists
#   CROSS=1 sbatch exp/mask.sh               # + the sent4/ear adapters
#   TAG=v2 sbatch exp/mask.sh                # keep an earlier run's files
#
# An adapter can be given as a comma-separated stack, merged left to right --
# needed for a DPO checkpoint from before dpo_qwen.py trained the SFT adapter
# in place, which is a delta on the SFT weights rather than on the base:
#   STAGES=dpo DPO_ADAPTER="$SFT_ADAPTER,$DPO_ADAPTER" sbatch exp/mask.sh
#
# A stage whose adapter is missing is skipped with a warning rather than
# failing the job -- the usual reason to run this is that one stage just
# finished training.

source ~/.bashrc
set -eo pipefail

DS_ID="${DS_ID:-keylazy/slurp-mask-v1}"
TAG="${TAG:-mask}"
STAGES="${STAGES:-base sft dpo}"
SFT_ADAPTER="${SFT_ADAPTER:-checkpoints/Qwen2.5-Omni-3B-mask-sft}"
DPO_ADAPTER="${DPO_ADAPTER:-checkpoints/Qwen2.5-Omni-3B-mask-dpo}"

mkdir -p results logs
conda activate qwen25omni

JUDGE_HOST=$(cat /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge_host.txt)
JUDGE_URL="http://${JUDGE_HOST}:8000/v1"
if ! curl -sf --max-time 10 "${JUDGE_URL}/models" > /dev/null; then
    echo "vLLM judge not reachable at ${JUDGE_URL} -- start it: sbatch /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge.slurm" >&2
    exit 1
fi
# take the judge name from the server: the box gets re-served with different
# models, and a mismatch is a 404 on every row, discovered ~7 min in
JUDGE_MODEL=$(curl -sf --max-time 10 "${JUDGE_URL}/models" \
    | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
echo "judge OK at ${JUDGE_URL} serving ${JUDGE_MODEL}"

run_eval() {
    local adapter="$1"; shift
    local label="${adapter:-Qwen/Qwen2.5-Omni-3B}"
    if [[ -n "$adapter" && ! -e "$adapter" && "$adapter" != */* ]]; then
        echo "!! skipping ${label}: no such adapter" >&2
        return 0
    fi
    echo "=== eval ${label##*/} on ${DS_ID} ==="
    python -u mask_eval_qwen.py \
        --dataset "$DS_ID" \
        --split test \
        --model-path Qwen/Qwen2.5-Omni-3B \
        ${adapter:+--adapter-path "$adapter"} \
        --num-rows -1 \
        --tag "$TAG" \
        --judge-base-url "$JUDGE_URL" \
        --judge-model "$JUDGE_MODEL" \
        "$@"
}

for stage in $STAGES; do
    case "$stage" in
        base) run_eval "" ;;
        sft)  [[ -d "$SFT_ADAPTER" ]] && run_eval "$SFT_ADAPTER" \
                  || echo "!! skipping sft: ${SFT_ADAPTER} not found" >&2 ;;
        dpo)  [[ -d "${DPO_ADAPTER##*,}" ]] && run_eval "$DPO_ADAPTER" \
                  || echo "!! skipping dpo: ${DPO_ADAPTER} not found" >&2 ;;
        *)    echo "!! unknown stage ${stage}" >&2 ;;
    esac
done

# cross-track transfer: the mask test split was built leak-free against both of
# these adapters' training sentences (mask_data.py --exclude-ds), which is what
# makes scoring them here legitimate
if [[ -n "${CROSS:-}" ]]; then
    run_eval keylazy/Qwen2.5-Omni-3B-bab-sent4-sft
    run_eval keylazy/Qwen2.5-Omni-3B-ear-sft-adapter
fi

echo "=== done, results in results/mask_results_*_${TAG}.jsonl ==="
