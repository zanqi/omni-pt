#!/bin/bash
source ~/.bashrc

# Dataset repo ids to push to. Override by passing them as args:
#   ./babble_data.sh [DS_ID_QWEN25] [DS_ID_QWEN3]
# Defaults match the datasets eval.sh consumes.
DS_ID_QWEN25="${1:-keylazy/slurp-babble-Qwen2.5-Omni-3B-v3}"
DS_ID_QWEN3="${2:-keylazy/slurp-babble-Qwen3-Omni-30B-A3B-Instruct-v1}"

# fail-fast: is the vLLM judge reachable? (babble_data.py reads the host file
# itself; this curl is only a precheck)
JUDGE_HOST=$(cat /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge_host.txt)
JUDGE_URL="http://${JUDGE_HOST}:8000/v1"
if ! curl -sf --max-time 10 "${JUDGE_URL}/models" > /dev/null; then
    echo "vLLM judge not reachable at ${JUDGE_URL} -- start it: sbatch /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge.slurm" >&2
    exit 1
fi

echo "Generating Qwen2.5 dataset -> ${DS_ID_QWEN25}"
conda activate qwen25omni
python -u babble_data.py --omni-path Qwen/Qwen2.5-Omni-3B --ds-id "$DS_ID_QWEN25"
echo "Qwen2.5 dataset generated successfully."

echo "Generating Qwen3 dataset -> ${DS_ID_QWEN3}"
conda activate qwen3omni
python -u babble_data.py --omni-path Qwen/Qwen3-Omni-30B-A3B-Instruct --ds-id "$DS_ID_QWEN3"
echo "Qwen3 dataset generated successfully."