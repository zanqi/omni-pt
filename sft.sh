#!/bin/bash
# LoRA SFT: one adapter per omni model, trained on the track's train split
# exactly as built -- the dataset's own answer:repair:repeat mix is what the
# model sees. Epochs are fixed at 3 (188 optimizer steps/epoch at 3K rows).
#
# Adapters land in checkpoints/<model>-bab[-<track>]-sft and push to
# keylazy/<model>-bab[-<track>]-sft.
#
# The answer-row composition sweep is gone with sft_qwen.py's --train-caps;
# MULTS is no longer read here (eval.sh still reads it, to pick which of the
# already-trained -<n>x adapters to score).
#
# Usage:
#   ./sft.sh              # both models, qwen2.5 family first
#   ./sft.sh qwen25       # only Qwen2.5-Omni-3B
#   ./sft.sh qwen3        # only Qwen3-Omni-30B-A3B-Instruct
#   HR=1 ./sft.sh         # retired: needed TASK_PROMPT_HR, which sft_qwen.py
#                         # no longer trains under
#   TREE=1 ./sft.sh       # intersect-the-two-passes track — tree-v1 datasets,
#                         # TASK_PROMPT_TREE (the restate prompt the data was
#                         # probed under), <model>-bab-tree-sft
#   BEAM=1 ./sft.sh       # retired: beam-v1 rows were probed under the plain
#                         # TASK_PROMPT, which sft_qwen.py no longer trains under
#   SENT2=1 ./sft.sh      # sent-loss, two witnesses — sent2-v1 datasets. Same
#                         # two probe passes as TREE, so the same restate
#                         # prompt, <model>-bab-sent2-sft
#   SENT4=1 ./sft.sh      # retired: sent4-v1 rows were probed under the plain
#                         # TASK_PROMPT, same as BEAM
#   CR=1 ./sft.sh         # C/R-only track — beam-v3 datasets with the repeat
#                         # rows dropped, adapter <model>-bab-cr-sft
#   EPOCHS=6 ADAPTER_KIND=bab-cr6 CR=1 ./sft.sh qwen25
#                         # step-matched CR: half the rows of a beam run, so
#                         # twice the epochs to land on the same 750 steps.
#                         # ADAPTER_KIND renames the output so it doesn't
#                         # overwrite the 3-epoch adapter; pass the same
#                         # ADAPTER_KIND to eval.sh to score it.
#
# One GPU, sequential — run it under salloc/srun or wrap it in an sbatch job.
source ~/.bashrc
set -eo pipefail

WHICH="${1:-both}"
EPOCHS="${EPOCHS:-3}"
# captured before the track case overwrites it, so a step-matched or otherwise
# renamed run can keep a prior adapter of the same track intact (mirrors eval.sh)
ADAPTER_KIND_ENV="${ADAPTER_KIND:-}"

# sft_qwen.py trains under the restate prompt only (prompts.get_prompts), so a
# track whose rows were probed under TASK_PROMPT_HR / the plain TASK_PROMPT has
# no way to match its data any more and stops here rather than training a
# mismatched adapter.
# extra per-track flags -- currently only CR's kind filter
EXTRA_FLAGS=""
if [[ -n "${HR:-}" ]]; then
    echo "HR track needs TASK_PROMPT_HR, which sft_qwen.py no longer trains under" >&2
    exit 1
elif [[ -n "${TREE:-}" ]]; then
    # restate prompt, which is sft_qwen.py's default -- no flag needed
    ADAPTER_KIND="bab-tree-adapter"
    DEFAULT_QWEN25="keylazy/slurp-babble-Qwen2.5-Omni-3B-tree-v1"
    DEFAULT_QWEN3="keylazy/slurp-babble-Qwen3-Omni-30B-A3B-Instruct-tree-v1"
elif [[ -n "${BEAM:-}" ]]; then
    echo "BEAM track needs the plain TASK_PROMPT, which sft_qwen.py no longer trains under" >&2
    exit 1
elif [[ -n "${SENT2:-}" ]]; then
    # same two probe passes as the tree track, so the same restate prompt,
    # which is sft_qwen.py's default -- no flag needed
    ADAPTER_KIND="bab-sent2-adapter"
    DEFAULT_QWEN25="keylazy/slurp-babble-Qwen2.5-Omni-3B-sent2-v1"
    DEFAULT_QWEN3="keylazy/slurp-babble-Qwen3-Omni-30B-A3B-Instruct-sent2-v1"
elif [[ -n "${SENT4:-}" ]]; then
    # no task-response pass on this track, so its rows were probed plain
    echo "SENT4 track needs the plain TASK_PROMPT, which sft_qwen.py no longer trains under" >&2
    exit 1
elif [[ -n "${CR:-}" ]]; then
    # C/R only: the beam-v3 rows as built, minus every repeat row. Same audio
    # and targets as the beam track, so the only variable is the removed third
    # dimension.
    EXTRA_FLAGS="--train-kinds answer,repair"
    ADAPTER_KIND="bab-cr-adapter"
    DEFAULT_QWEN25="keylazy/slurp-babble-Qwen2.5-Omni-3B-beam-v3"
    DEFAULT_QWEN3=""  # no qwen3 beam dataset exists; pass DS_QWEN3 explicitly
else
    ADAPTER_KIND="bab-adapter"
    DEFAULT_QWEN25="keylazy/slurp-babble-Qwen2.5-Omni-3B-v4"
    DEFAULT_QWEN3="keylazy/slurp-babble-Qwen3-Omni-30B-A3B-Instruct-v2"
fi
[[ -n "$ADAPTER_KIND_ENV" ]] && ADAPTER_KIND="$ADAPTER_KIND_ENV"
DS_QWEN25="${DS_QWEN25:-$DEFAULT_QWEN25}"
DS_QWEN3="${DS_QWEN3:-$DEFAULT_QWEN3}"

run_sft() {
    local omni_path="$1" ds_id="$2"
    local model_name="${omni_path##*/}"
    local run_name

    if [[ -z "$ds_id" ]]; then
        echo "no dataset for ${model_name} on this track — set DS_QWEN25/DS_QWEN3" >&2
        exit 1
    fi

    run_name="${model_name}-${ADAPTER_KIND%-adapter}-sft"
    echo "=== ${run_name}: train split as built, ${EPOCHS} epochs <- ${ds_id} ==="
    python -u sft_qwen.py $EXTRA_FLAGS \
        --omni-path "$omni_path" \
        --repair-ds-id "$ds_id" \
        --repair-repo-name "$run_name" \
        --repair-epochs "$EPOCHS"
    echo "=== ${run_name} done ==="
}

if [[ "$WHICH" == "both" || "$WHICH" == "qwen25" ]]; then
    conda activate qwen25omni
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    run_sft Qwen/Qwen2.5-Omni-3B "$DS_QWEN25"
fi

if [[ "$WHICH" == "both" || "$WHICH" == "qwen3" ]]; then
    conda activate qwen3omni
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    run_sft Qwen/Qwen3-Omni-30B-A3B-Instruct "$DS_QWEN3"
fi
