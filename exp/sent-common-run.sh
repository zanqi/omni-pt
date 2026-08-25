#!/bin/bash
# The body of exp/sent-common.sh's job -- the CRF-style comparison of five
# models on ONE common test set, run in-band so the sbatch --wrap stays
# readable.
#
# The existing sent tables (results/viz.ipynb, "CRF tables for sent-based key
# pieces runs") score each adapter on its OWN split, so base(sent2) and
# base(sent4) are different numbers and the two SFT columns are not directly
# comparable. Here every row is scored on sent2's test split:
#
#   base, sent2 SFT, sent4 SFT, the white-noise (EAR-track) adapter, and the
#   mask-track DPO checkpoint.
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
# Both passes enforce restatement: an `answer` reply earns full credit only if
# it names the command's specifics back, which is the evidence the audio got
# through -- the only thing this benchmark measures. So the type-judge pass
# uses RESPONSE_TYPE_SYSTEM (no --no-restate-judge), and the per-kind pass uses
# the post-375dc9b ANSWER_JUDGE_SYSTEM, which drops a right-but-vague reply to
# 0.5. This is the judge for the track going forward.
#
# That makes every row here incomparable to the sent-common files produced on
# 2026-08-24, which were scored before the ANSWER_JUDGE_SYSTEM change and with
# --no-restate-judge -- all five rows must be re-run together, which is why
# MODELS defaults to all of them.
#
# Prompts stay train-matched: only sent4 was trained without the restatement
# clause, so it alone runs --plain-prompt. Everything else -- including the
# untrained base, which the judge is likewise entitled to ask for a
# restatement from -- takes TASK_PROMPT_TREE, now babble_eval_qwen.py's
# default, which is why only sent4 carries a prompt flag at all.
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

    echo "=== eval ${name} (${prompt_flag:-restate prompt}, ${tag}) on ${DS} -> ${out} ==="
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
        "--score-matrix tree"
}

# One table row each, "<selector>|<adapter>|<prompt flag>". MODELS names the
# subset to run -- useful for retrying a single row after a crash, but note
# that a table mixing rows from two judge revisions is invalid, so the default
# is all of them.
ROWS=(
    "base||"
    "sent2|keylazy/Qwen2.5-Omni-3B-bab-sent2-sft|"
    # the one adapter whose targets never restate, so it keeps TASK_PROMPT
    "sent4|keylazy/Qwen2.5-Omni-3B-bab-sent4-sft|--plain-prompt"
    # out-of-track reference: SFT'd on the white-noise EAR split
    "ear|keylazy/Qwen2.5-Omni-3B-ear-sft-adapter|"
    # mask track, SFT then DPO on the balanced prefs
    "mask-dpo|keylazy/Qwen2.5-Omni-3B-mask-dpo-bal|"
)
MODELS="${MODELS:-base sent2 sent4 ear mask-dpo}"

for row in "${ROWS[@]}"; do
    IFS='|' read -r selector adapter prompt_flag <<< "$row"
    [[ " ${MODELS} " == *" ${selector} "* ]] || continue
    run_both_judges "$adapter" "$prompt_flag"
done
