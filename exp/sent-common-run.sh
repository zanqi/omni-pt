#!/bin/bash
# The body of exp/sent-common.sh's job -- the CRF-style comparison of four
# models on ONE common test set, run in-band so the sbatch --wrap stays
# readable.
#
# The existing sent tables (results/viz.ipynb, "CRF tables for sent-based key
# pieces runs") score each adapter on its OWN split, so base(sent2) and
# base(sent4) are different numbers and the two SFT columns are not directly
# comparable. Here every row is scored on sent2's test split:
#
#   base, sent2 SFT, sent4 SFT, and the white-noise (EAR-track) adapter.
#
# Kinds are answer,repair only -- EAR_2 = 2*C*R/(C+R). The ear adapter's track
# has no `repeat` rows, so including F would score it on a behavior it was
# never trained for.
#
# Each model is scored twice off the same generation rubric pair, as in
# exp/sent-2.sh: the per-kind judges (tag `sent-common`) and the type judge +
# tree matrix (tag `sent-common-typejudge`). Numbers are comparable within a
# judge only.
#
# Prompts are train-matched: only sent2 was trained to restate what it heard,
# so it alone runs --restate-prompt; base, sent4 and ear run the plain
# TASK_PROMPT. That asymmetry is also why the type-judge pass adds
# --no-restate-judge: RESPONSE_TYPE_SYSTEM types a reply "answer" only if its
# wording accounts for every key element the command spoke, which the three
# plain-prompt models cannot satisfy by construction. The per-kind pass needs
# no such flag -- ANSWER_JUDGE_SYSTEM already credits a reply that simply acts
# on the command.
source ~/.bashrc
set -eo pipefail

conda activate qwen25omni

DS=keylazy/slurp-babble-Qwen2.5-Omni-3B-sent2-v1
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

# both judge passes over one model, so a model is loaded once per pass and the
# two result files always cover the same four rows
run_eval() {
    local adapter_path="$1" prompt_flag="$2" tag="$3" judge_flags="$4"
    local name="${adapter_path:-Qwen/Qwen2.5-Omni-3B}"
    name="${name##*/}"
    local out="${OUT_DIR}/bab_${name}_${tag}.jsonl"

    echo "=== eval ${name} (${prompt_flag:-plain prompt}, ${tag}) on ${DS} -> ${out} ==="
    python -u babble_eval_qwen.py $judge_flags \
        --model-path Qwen/Qwen2.5-Omni-3B \
        ${adapter_path:+--adapter-path "$adapter_path"} \
        ${prompt_flag} \
        --dataset "$DS" \
        --split test \
        --num-rows -1 \
        --kinds answer,repair \
        --out "$out" \
        --judge-base-url "$JUDGE_URL" \
        --judge-model "$JUDGE_MODEL"
}

run_both_judges() {
    local adapter_path="$1" prompt_flag="$2"
    run_eval "$adapter_path" "$prompt_flag" sent-common "--judge-mode per-kind"
    run_eval "$adapter_path" "$prompt_flag" sent-common-typejudge \
        "--score-matrix tree --no-restate-judge"
}

run_both_judges "" ""
run_both_judges keylazy/Qwen2.5-Omni-3B-bab-sent2-sft --restate-prompt
run_both_judges keylazy/Qwen2.5-Omni-3B-bab-sent4-sft ""
# out-of-track reference: SFT'd on the white-noise EAR split, plain prompt
run_both_judges keylazy/Qwen2.5-Omni-3B-ear-sft-adapter ""
