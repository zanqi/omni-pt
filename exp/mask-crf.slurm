#!/bin/bash

#SBATCH --job-name=mask_crf
#SBATCH --account=sciencehub
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=188G
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt
#SBATCH --output=logs/mask_crf_%j.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zanqil@uw.edu

# exp/mask-crf.slurm -- the three-kind mask track end to end (steps/mask.html).
#
#   ./exp/mask-crf.slurm        # here, in a shell you hold
#   sbatch exp/mask-crf.slurm   # queued, same file
#
# One job rather than five: the stages are strictly sequential (each trains on
# what the last one produced), so splitting them only adds four queue waits,
# and the 16 CPUs the build wants are the binding request. --time is 2 days for
# the whole track; a job that only holds the GPU as long as it needs costs
# nothing extra.

source ~/.bashrc
set -eo pipefail
mkdir -p results logs

CFG=configs/mask-crf.yaml
SFT=checkpoints/Qwen2.5-Omni-3B-mask-v2-sft
DPO=checkpoints/Qwen2.5-Omni-3B-mask-v2-dpo

# the build alone runs in qwen3omni: that is where transformers is new enough
# to carry the forced aligner, and openai-whisper (the mask gate) is installed
# beside it. Everything after it loads an omni model, which wants qwen25omni.
conda activate qwen3omni
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python -u mask_data.py --config $CFG

conda activate qwen25omni
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python -u sft_qwen.py      --config $CFG
python -u mask_dpo_data.py --config $CFG
python -u dpo_qwen.py      --config $CFG

# one test split, one judge, three lines -- base is what makes the SFT and DPO
# numbers mean anything. The adapter is a flag, not a config key: which model a
# row evaluates is this driver's decision.
python -u mask_eval_qwen.py --config $CFG
python -u mask_eval_qwen.py --config $CFG --adapter-path $SFT
python -u mask_eval_qwen.py --config $CFG --adapter-path $DPO

# cross-track transfer, if wanted: the mask test split was built leak-free
# against both of these adapters' training sentences (mask_data.py
# --exclude-ds), which is what makes scoring them here legitimate.
# python -u mask_eval_qwen.py --config $CFG --adapter-path keylazy/Qwen2.5-Omni-3B-bab-sent4-sft
# python -u mask_eval_qwen.py --config $CFG --adapter-path keylazy/Qwen2.5-Omni-3B-ear-sft-adapter
