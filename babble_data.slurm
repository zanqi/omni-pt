#!/bin/bash

#SBATCH --job-name=babble_data
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zanqil@uw.edu

#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=90G
#SBATCH --gpus=1
#SBATCH --time=01-0:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt
#SBATCH --output=babble_data_%j.log
#SBATCH --export=all
source ~/.bashrc
conda activate qwen3omni
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# fail fast if the vLLM judge (target-generation) server isn't reachable
JUDGE_HOST=$(cat /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge_host.txt)
if ! curl -sf --max-time 10 "http://${JUDGE_HOST}:8000/v1/models" > /dev/null; then
    echo "ERROR: vLLM judge at http://${JUDGE_HOST}:8000 is not responding." >&2
    echo "Start it first: sbatch /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge.slurm" >&2
    exit 1
fi
echo "judge OK at http://${JUDGE_HOST}:8000/v1"

# -u: unbuffered stdout so errors/progress show up in the log immediately
python -u babble_data.py --omni-path Qwen/Qwen3-Omni-30B-A3B-Instruct

# sbatch /gscratch/sciencehub/zanqil/projects/omni-pt/babble_data.slurm