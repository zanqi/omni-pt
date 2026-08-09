#!/bin/bash
# LoRA SFT sweep: for each omni model, train four adapters that differ only in
# how many `answer` rows the train split carries.
#
#   1x -> 1K:1K:1K   2x -> 2K:1K:1K   3x -> 3K:1K:1K   4x -> 4K:1K:1K
#          (answer : repair : repeat)
#
# repair/repeat are capped at the 1K the datasets contain, so 1x is exactly the
# 3K interleaved triplet block and 4x is the full 6K train split. Epochs are
# fixed at 3, so optimizer steps grow with the data (188/epoch at 3K rows).
#
# Adapters land in ./<model>-bab-adapter-<n>x and push to
# keylazy/<model>-bab-adapter-<n>x.
#
# Usage:
#   ./sft.sh              # all 8 runs, qwen2.5 family first
#   ./sft.sh qwen25       # only the Qwen2.5-Omni-3B sweep
#   ./sft.sh qwen3        # only the Qwen3-Omni-30B-A3B-Instruct sweep
#   MULTS="3 4" ./sft.sh  # only the 3x and 4x runs
#   MULTS= ./sft.sh       # no sweep: ONE run on the whole train split, whatever
#                         # its composition -> adapter <model>-bab[-<track>]-sft
#   HR=1 ./sft.sh         # Heard:/Reply: track — hr-v1 datasets, TASK_PROMPT_HR,
#                         # adapters named <model>-bab-hr-adapter-<n>x
#   TREE=1 ./sft.sh       # intersect-the-two-passes track — tree-v1 datasets,
#                         # TASK_PROMPT_TREE (the restate prompt the data was
#                         # probed under), <model>-bab-tree-adapter-<n>x
#   BEAM=1 ./sft.sh       # beam-consensus track — beam-v1 datasets, plain
#                         # TASK_PROMPT, adapters <model>-bab-beam-adapter-<n>x
#   CR=1 ./sft.sh         # C/R-only track — beam-v3 datasets with the repeat
#                         # rows dropped and answer capped to match repair
#                         # (1K:1K), adapters <model>-bab-cr[-adapter-<n>x]
#   EPOCHS=6 ADAPTER_KIND=bab-cr6-adapter CR=1 MULTS= ./sft.sh qwen25
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
MULTS="${MULTS-1 2 3 4}"  # no colon: MULTS= means "single run", not "default"
EPOCHS="${EPOCHS:-3}"
# captured before the track case overwrites it, so a step-matched or otherwise
# renamed run can keep a prior adapter of the same track intact (mirrors eval.sh)
ADAPTER_KIND_ENV="${ADAPTER_KIND:-}"

# which task prompt the adapter trains under -- it must match the one the
# dataset's probes and targets were built under
PROMPT_FLAG=""
# extra per-track flags (currently only CR's kind filter) and the sweep's
# non-answer caps, which CR shrinks to 'repair' alone
EXTRA_FLAGS=""
FIXED_CAPS="repair=1000,repeat=1000"
if [[ -n "${HR:-}" ]]; then
    PROMPT_FLAG="--heard-reply"
    ADAPTER_KIND="bab-hr-adapter"
    DEFAULT_QWEN25="keylazy/slurp-babble-Qwen2.5-Omni-3B-hr-v1"
    DEFAULT_QWEN3="keylazy/slurp-babble-Qwen3-Omni-30B-A3B-Instruct-hr-v1"
elif [[ -n "${TREE:-}" ]]; then
    PROMPT_FLAG="--restate-prompt"
    ADAPTER_KIND="bab-tree-adapter"
    DEFAULT_QWEN25="keylazy/slurp-babble-Qwen2.5-Omni-3B-tree-v1"
    DEFAULT_QWEN3="keylazy/slurp-babble-Qwen3-Omni-30B-A3B-Instruct-tree-v1"
elif [[ -n "${BEAM:-}" ]]; then
    # beam rows were probed under the plain TASK_PROMPT, so no prompt flag
    ADAPTER_KIND="bab-beam-adapter"
    DEFAULT_QWEN25="keylazy/slurp-babble-Qwen2.5-Omni-3B-beam-v1"
    DEFAULT_QWEN3="keylazy/slurp-babble-Qwen3-Omni-30B-A3B-Instruct-beam-v1"
elif [[ -n "${CR:-}" ]]; then
    # C/R only: the beam-v3 rows as built, minus every repeat row, with answer
    # cut from 2K to the 1K that matches repair. Same audio and targets as the
    # beam track, so the only variable is the removed third dimension.
    EXTRA_FLAGS="--kinds answer,repair"
    FIXED_CAPS="repair=1000"
    SINGLE_CAPS="answer=1000,repair=1000"
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
# caps for the single-run path: empty for every track but CR, which must still
# cut the surplus answer rows even when it isn't sweeping the mix
SINGLE_CAPS="${SINGLE_CAPS:-}"

run_sweep() {
    local omni_path="$1" ds_id="$2"
    local model_name="${omni_path##*/}"
    local n run_name

    if [[ -z "$ds_id" ]]; then
        echo "no dataset for ${model_name} on this track — set DS_QWEN25/DS_QWEN3" >&2
        exit 1
    fi

    if [[ -z "$MULTS" ]]; then
        # train on the split as built -- no --train-caps unless the track needs
        # one, so the dataset's own answer:repair:repeat mix is what the model sees
        run_name="${model_name}-${ADAPTER_KIND%-adapter}-sft"
        echo "=== ${run_name}: train split ${SINGLE_CAPS:-as built}, ${EPOCHS} epochs <- ${ds_id} ==="
        python -u sft_qwen.py $PROMPT_FLAG $EXTRA_FLAGS \
            --omni-path "$omni_path" \
            --ds-id "$ds_id" \
            ${SINGLE_CAPS:+--train-caps "$SINGLE_CAPS"} \
            --run-name "$run_name" \
            --epochs "$EPOCHS"
        echo "=== ${run_name} done ==="
        return
    fi

    for n in $MULTS; do
        run_name="${model_name}-${ADAPTER_KIND}-${n}x"
        echo "=== ${run_name}: answer=$((n * 1000)) ${FIXED_CAPS}, ${EPOCHS} epochs <- ${ds_id} ==="
        python -u sft_qwen.py $PROMPT_FLAG $EXTRA_FLAGS \
            --omni-path "$omni_path" \
            --ds-id "$ds_id" \
            --train-caps "answer=$((n * 1000)),${FIXED_CAPS}" \
            --run-name "$run_name" \
            --epochs "$EPOCHS"
        echo "=== ${run_name} done ==="
    done
}

if [[ "$WHICH" == "both" || "$WHICH" == "qwen25" ]]; then
    conda activate qwen25omni
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    run_sweep Qwen/Qwen2.5-Omni-3B "$DS_QWEN25"
fi

if [[ "$WHICH" == "both" || "$WHICH" == "qwen3" ]]; then
    conda activate qwen3omni
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    run_sweep Qwen/Qwen3-Omni-30B-A3B-Instruct "$DS_QWEN3"
fi
