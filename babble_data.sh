#!/bin/bash
source ~/.bashrc

# Build the babble dataset for one or both omni families. Defaults match the
# datasets eval.sh consumes.
#
# Usage (same shape as sft.sh / eval.sh):
#   ./babble_data.sh                 # both families, qwen2.5 first
#   ./babble_data.sh qwen25          # only Qwen2.5-Omni-3B
#   ./babble_data.sh qwen3           # only Qwen3-Omni-30B-A3B-Instruct
#   DS_QWEN25=me/my-ds ./babble_data.sh qwen25   # override the pushed repo id
#   N_EXTRA_ANS=1000 ./babble_data.sh qwen25     # row counts (see --help for
#   N_TRAIN=1000 N_TEST=50 ./babble_data.sh        defaults)
#
#   HR=1 ./babble_data.sh    # the Heard:/Reply: track (one probe pass, few-shot
#                            # labeler, disjoint SNR bands) -> ...-hr-v1 ids
#   TREE=1 ./babble_data.sh  # the intersection track (same two probe passes,
#                            # each labeled independently for what it lost, the
#                            # two loss lists intersected) -> ...-tree-v1 ids
#   BEAM=1 ./babble_data.sh  # the beam-consensus track (one N-best ASR pass, no
#                            # task-response pass) -> ...-beam-v1 ids
set -eo pipefail

WHICH="${1:-both}"
if [[ -n "${HR:-}" ]]; then
    TRACK_FLAG="--heard-reply"
    DEFAULT_QWEN25="keylazy/slurp-babble-Qwen2.5-Omni-3B-hr-v1"
    DEFAULT_QWEN3="keylazy/slurp-babble-Qwen3-Omni-30B-A3B-Instruct-hr-v1"
elif [[ -n "${TREE:-}" ]]; then
    TRACK_FLAG="--tree-label"
    DEFAULT_QWEN25="keylazy/slurp-babble-Qwen2.5-Omni-3B-tree-v1"
    DEFAULT_QWEN3="keylazy/slurp-babble-Qwen3-Omni-30B-A3B-Instruct-tree-v1"
elif [[ -n "${BEAM:-}" ]]; then
    TRACK_FLAG="--beam-label"
    DEFAULT_QWEN25="keylazy/slurp-babble-Qwen2.5-Omni-3B-beam-v1"
    DEFAULT_QWEN3="keylazy/slurp-babble-Qwen3-Omni-30B-A3B-Instruct-beam-v1"
else
    TRACK_FLAG=""
    DEFAULT_QWEN25="keylazy/slurp-babble-Qwen2.5-Omni-3B-v3"
    DEFAULT_QWEN3="keylazy/slurp-babble-Qwen3-Omni-30B-A3B-Instruct-v1"
fi
DS_QWEN25="${DS_QWEN25:-$DEFAULT_QWEN25}"
DS_QWEN3="${DS_QWEN3:-$DEFAULT_QWEN3}"

# row counts: unset means babble_data.py's own defaults
COUNTS=""
[[ -n "${N_TRAIN:-}" ]]     && COUNTS="$COUNTS --n-train $N_TRAIN"
[[ -n "${N_TEST:-}" ]]      && COUNTS="$COUNTS --n-test $N_TEST"
[[ -n "${N_EXTRA_ANS:-}" ]] && COUNTS="$COUNTS --n-extra-ans $N_EXTRA_ANS"

# fail-fast: is the vLLM judge reachable? (babble_data.py reads the host file
# itself; this curl is only a precheck)
JUDGE_HOST=$(cat /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge_host.txt)
JUDGE_URL="http://${JUDGE_HOST}:8000/v1"
if ! curl -sf --max-time 10 "${JUDGE_URL}/models" > /dev/null; then
    echo "vLLM judge not reachable at ${JUDGE_URL} -- start it: sbatch /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge.slurm" >&2
    exit 1
fi

if [[ "$WHICH" == "both" || "$WHICH" == "qwen25" ]]; then
    echo "Generating Qwen2.5 dataset -> ${DS_QWEN25}"
    conda activate qwen25omni
    python -u babble_data.py $TRACK_FLAG $COUNTS --omni-path Qwen/Qwen2.5-Omni-3B --ds-id "$DS_QWEN25"
    echo "Qwen2.5 dataset generated successfully."
fi

if [[ "$WHICH" == "both" || "$WHICH" == "qwen3" ]]; then
    echo "Generating Qwen3 dataset -> ${DS_QWEN3}"
    conda activate qwen3omni
    python -u babble_data.py $TRACK_FLAG $COUNTS --omni-path Qwen/Qwen3-Omni-30B-A3B-Instruct --ds-id "$DS_QWEN3"
    echo "Qwen3 dataset generated successfully."
fi