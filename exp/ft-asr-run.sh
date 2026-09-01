#!/bin/bash

# exp/ft-asr-run.sh -- the body of exp/ft-asr.sh's job, in-band so the
# sbatch --wrap stays readable (same shape as exp/sent-common-run.sh).
#
# It also has to be its own #!/bin/bash file: sbatch --wrap runs its script
# under sh, where `source ~/.bashrc` dies in ~/.fzf.bash's process
# substitution before conda is ever on PATH.
source ~/.bashrc
set -eo pipefail

CFG=configs/ft-asr.yaml
conda activate qwen25omni
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
mkdir -p results logs

if [[ "$PHASE" == "data" ]]; then
    python -u asr_data.py --config "$CFG"

    # --task asr must be passed: the config file serves both of this script's
    # sft_qwen.py runs, and `task` is deliberately not spelled in it -- the flag
    # is what picks which half of the repair_*/asr_* keys the script reads.
    python -u sft_qwen.py --task asr --config "$CFG"

    # the gate (steps/ft-asr-2.html steps 7-8). Four runs: base and ft on the
    # ASR test split, then both again on the sent4-v1 test rows -- different
    # SLURP split and SNR draws, so that pair is the generalization check on
    # audio resembling what babble_data.py will feed the labeler. Those rows
    # keep the transcript in `sentence` (their `target` is a repair reply),
    # hence --ref-column.
    SENT4_DS=keylazy/slurp-babble-Qwen2.5-Omni-3B-sent4-v1
    python -u asr_eval_qwen.py --config "$CFG" --tag base
    python -u asr_eval_qwen.py --config "$CFG" --tag ft \
        --adapter-path checkpoints/Qwen2.5-Omni-3B-asr-sft
    python -u asr_eval_qwen.py --config "$CFG" --tag base-on-sent4 \
        --asr-ds-id "$SENT4_DS" --split test --ref-column sentence
    python -u asr_eval_qwen.py --config "$CFG" --tag ft-on-sent4 \
        --adapter-path checkpoints/Qwen2.5-Omni-3B-asr-sft \
        --asr-ds-id "$SENT4_DS" --split test --ref-column sentence
    echo "=== gate: compare results/asr_*_base.jsonl and results/asr_*_ft.jsonl ==="
    exit 0
fi

# --- sent4 phase: the labeler and the judge both need the vLLM box ---
JUDGE_HOST=$(cat /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge_host.txt)
JUDGE_URL="http://${JUDGE_HOST}:8000/v1"
if ! curl -sf --max-time 10 "${JUDGE_URL}/models" > /dev/null; then
    echo "vLLM not reachable at ${JUDGE_URL} -- sbatch /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge.slurm" >&2
    exit 1
fi
# the judge box gets re-served with different models, so take the name from the
# server rather than trusting a default -- a mismatch is a 404 on every row
JUDGE_MODEL=$(curl -sf --max-time 10 "${JUDGE_URL}/models" \
    | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
echo "judge OK at ${JUDGE_URL} serving ${JUDGE_MODEL}"

python -u babble_data.py --config "$CFG"
python -u sft_qwen.py --config "$CFG"

# Two judge passes off the one adapter, as on the sent tracks: the per-kind
# judges (comparable with the beam rows) and the type judge + tree matrix
# (comparable with the tree rows). Numbers are comparable within a judge only.
# Base and adapter both run, so the delta is measured on this dataset rather
# than read across from the v1 table.
for adapter in "" "checkpoints/Qwen2.5-Omni-3B-bab-sent4ft-sft"; do
    name="${adapter:-Qwen/Qwen2.5-Omni-3B}"; name="${name##*/}"
    for pass in "per-kind|sent4-ftasr" "tree|sent4-ftasr-typejudge"; do
        IFS='|' read -r mode tag <<< "$pass"
        flags="--judge-mode per-kind"
        [[ "$mode" == tree ]] && flags="--score-matrix tree"
        echo "=== eval ${name} (${tag}) ==="
        python -u babble_eval_qwen.py --config "$CFG" $flags \
            ${adapter:+--adapter-path "$adapter"} \
            --out "results/bab_results_${name}_${tag}.jsonl" \
            --judge-base-url "$JUDGE_URL" --judge-model "$JUDGE_MODEL"
    done
done
