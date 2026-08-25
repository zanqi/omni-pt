#!/bin/bash
# The body of exp/white-cr.sh's job -- four evals of the white-noise C/R
# split, run in-band so the sbatch --wrap stays readable.
source ~/.bashrc
set -eo pipefail

conda activate qwen25omni

DS=keylazy/slurp-ear-sft
TAG=white-cr
OUT_DIR=results
mkdir -p "$OUT_DIR" logs

JUDGE_HOST=$(cat /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge_host.txt)
JUDGE_URL="http://${JUDGE_HOST}:8000/v1"
if ! curl -sf --max-time 10 "${JUDGE_URL}/models" > /dev/null; then
    echo "vLLM judge not reachable at ${JUDGE_URL} — start it: sbatch /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge.slurm" >&2
    exit 1
fi
# The judge box gets re-served with different models, so take the name from
# the server rather than trusting babble_eval_qwen.py's --judge-model default
# -- a mismatch is a 404 on every row, ~7 min in.
JUDGE_MODEL=$(curl -sf --max-time 10 "${JUDGE_URL}/models" \
    | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
echo "judge OK at ${JUDGE_URL} serving ${JUDGE_MODEL}"

run_eval() {
    local adapter_path="$1" prompt_flag="$2"
    local name="${adapter_path:-Qwen/Qwen2.5-Omni-3B}"
    name="${name##*/}"
    local out="${OUT_DIR}/bab_results_${name}_${TAG}.jsonl"

    echo "=== eval ${name} (${prompt_flag:-restate prompt}) on ${DS} -> ${out} ==="
    python -u babble_eval_qwen.py \
        --model-path Qwen/Qwen2.5-Omni-3B \
        ${adapter_path:+--adapter-path "$adapter_path"} \
        ${prompt_flag} \
        --dataset "$DS" \
        --split test \
        --num-rows -1 \
        --judge-mode per-kind \
        --kinds answer,repair \
        --out "$out" \
        --judge-base-url "$JUDGE_URL" \
        --judge-model "$JUDGE_MODEL"
}

# TASK_PROMPT_TREE is babble_eval_qwen.py's default, so the rows that ran
# under the plain prompt now have to say so -- an empty flag no longer means
# plain. Only sent2 was trained to restate.
run_eval "" --plain-prompt
run_eval keylazy/Qwen2.5-Omni-3B-bab-sent2-sft ""
run_eval keylazy/Qwen2.5-Omni-3B-bab-sent4-sft --plain-prompt
# in-domain reference: SFT'd on this same white-noise track, plain prompt
run_eval keylazy/Qwen2.5-Omni-3B-ear-sft-adapter --plain-prompt
