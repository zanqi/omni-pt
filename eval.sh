#!/bin/bash
# Babble-track eval sweep over the data-composition adapters trained by sft.sh.
#
# Per family: the base model, then the four adapters that differ only in how
# many `answer` rows they saw (1x -> 1K:1K:1K ... 4x -> 4K:1K:1K), all judged
# on the same 150-row test split of that family's dataset. 10 runs total.
#
# Results: results/bab_results_<model-or-adapter-name>_v4.jsonl
#          (base -> ..._Qwen2.5-Omni-3B_v4.jsonl,
#           1x   -> ..._Qwen2.5-Omni-3B-bab-adapter-1x_v4.jsonl)
#
# Usage:
#   ./eval.sh                     # all 10 runs
#   ./eval.sh qwen25              # one family
#   MULTS="4" ./eval.sh qwen3     # base + the 4x adapter only
#   ADAPTER_PREFIX=keylazy ./eval.sh   # eval the pushed hub adapters, not ./<dir>
source ~/.bashrc
set -eo pipefail

WHICH="${1:-both}"
MULTS="${MULTS:-1 2 3 4}"
DS_QWEN25="${DS_QWEN25:-keylazy/slurp-babble-Qwen2.5-Omni-3B-v4}"
DS_QWEN3="${DS_QWEN3:-keylazy/slurp-babble-Qwen3-Omni-30B-A3B-Instruct-v2}"
# Where the adapters live: "." for sft.sh's local output dirs, "keylazy" for the
# pushed hub copies. Either way the result filename uses the adapter basename.
ADAPTER_PREFIX="${ADAPTER_PREFIX:-.}"
OUT_DIR="${OUT_DIR:-results}"
TAG="${TAG:-v4}"

JUDGE_HOST=$(cat /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge_host.txt)
JUDGE_URL="http://${JUDGE_HOST}:8000/v1"
if ! curl -sf --max-time 10 "${JUDGE_URL}/models" > /dev/null; then
    echo "vLLM judge not reachable at ${JUDGE_URL} — start it: sbatch /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge.slurm" >&2
    exit 1
fi
echo "judge OK at ${JUDGE_URL}"
mkdir -p "$OUT_DIR"

run_eval() {
    local model_path="$1" ds_id="$2" adapter_path="${3:-}"
    local name="${adapter_path:-$model_path}"
    name="${name##*/}"
    local out="${OUT_DIR}/bab_results_${name}_${TAG}.jsonl"

    echo "=== eval ${name} on ${ds_id} -> ${out} ==="
    python -u babble_eval_qwen.py \
        --model-path "$model_path" \
        ${adapter_path:+--adapter-path "$adapter_path"} \
        --dataset "$ds_id" \
        --out "$out" \
        --judge-base-url "$JUDGE_URL"
}

run_family() {
    local model_path="$1" ds_id="$2"
    local model_name="${model_path##*/}" n

    run_eval "$model_path" "$ds_id"
    for n in $MULTS; do
        run_eval "$model_path" "$ds_id" "${ADAPTER_PREFIX}/${model_name}-bab-adapter-${n}x"
    done
}

if [[ "$WHICH" == "both" || "$WHICH" == "qwen25" ]]; then
    conda activate qwen25omni
    run_family Qwen/Qwen2.5-Omni-3B "$DS_QWEN25"
fi

if [[ "$WHICH" == "both" || "$WHICH" == "qwen3" ]]; then
    conda activate qwen3omni
    run_family Qwen/Qwen3-Omni-30B-A3B-Instruct "$DS_QWEN3"
fi