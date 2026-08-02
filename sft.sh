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
#   TREE=1 ./sft.sh       # decision-table track — tree-v1 datasets, plain
#                         # TASK_PROMPT, adapters <model>-bab-tree-adapter-<n>x
#
# One GPU, sequential — run it under salloc/srun or wrap it in an sbatch job.
source ~/.bashrc
set -eo pipefail

WHICH="${1:-both}"
MULTS="${MULTS-1 2 3 4}"  # no colon: MULTS= means "single run", not "default"

# the tree track trains under the plain prompt, so only HR passes a flag
HR_FLAG=""
if [[ -n "${HR:-}" ]]; then
    HR_FLAG="--heard-reply"
    ADAPTER_KIND="bab-hr-adapter"
    DEFAULT_QWEN25="keylazy/slurp-babble-Qwen2.5-Omni-3B-hr-v1"
    DEFAULT_QWEN3="keylazy/slurp-babble-Qwen3-Omni-30B-A3B-Instruct-hr-v1"
elif [[ -n "${TREE:-}" ]]; then
    ADAPTER_KIND="bab-tree-adapter"
    DEFAULT_QWEN25="keylazy/slurp-babble-Qwen2.5-Omni-3B-tree-v1"
    DEFAULT_QWEN3="keylazy/slurp-babble-Qwen3-Omni-30B-A3B-Instruct-tree-v1"
else
    ADAPTER_KIND="bab-adapter"
    DEFAULT_QWEN25="keylazy/slurp-babble-Qwen2.5-Omni-3B-v4"
    DEFAULT_QWEN3="keylazy/slurp-babble-Qwen3-Omni-30B-A3B-Instruct-v2"
fi
DS_QWEN25="${DS_QWEN25:-$DEFAULT_QWEN25}"
DS_QWEN3="${DS_QWEN3:-$DEFAULT_QWEN3}"

run_sweep() {
    local omni_path="$1" ds_id="$2"
    local model_name="${omni_path##*/}"
    local n run_name

    if [[ -z "$MULTS" ]]; then
        # train on the split exactly as built -- no --train-caps, so the
        # dataset's own answer:repair:repeat mix is what the model sees
        run_name="${model_name}-${ADAPTER_KIND%-adapter}-sft"
        echo "=== ${run_name}: full train split <- ${ds_id} ==="
        python -u sft_qwen.py $HR_FLAG \
            --omni-path "$omni_path" \
            --ds-id "$ds_id" \
            --run-name "$run_name" \
            --epochs 3
        echo "=== ${run_name} done ==="
        return
    fi

    for n in $MULTS; do
        run_name="${model_name}-${ADAPTER_KIND}-${n}x"
        echo "=== ${run_name}: answer=$((n * 1000)) repair=1000 repeat=1000 <- ${ds_id} ==="
        python -u sft_qwen.py $HR_FLAG \
            --omni-path "$omni_path" \
            --ds-id "$ds_id" \
            --train-caps "answer=$((n * 1000)),repair=1000,repeat=1000" \
            --run-name "$run_name" \
            --epochs 3
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
