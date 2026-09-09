#!/bin/bash

#SBATCH --job-name=bab_hr
#SBATCH --account=intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=188G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt
#SBATCH --output=logs/bab_hr_%j.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zanqil@uw.edu

# exp/bab-hr.slurm -- the heard-reply v2 track end to end (steps/bab-ds.html).
#
#   ./exp/bab-hr.slurm        # here, in a shell you hold
#   sbatch exp/bab-hr.slurm   # queued, same file
#
# Everything that varies lives in the config. To skip a stage, comment out its
# line; to redo one, paste it into a shell after the conda activate. The vLLM
# judge box is resolved inside the python (VLLM_HOST_FILE), so there is nothing
# to pass and nothing to precheck here.
#
# intelligentsystems/gpu-l40s, as babble_data_qwen25.slurm used for this same
# build: sciencehub/gpu-a40 is 52 CPUs total and the vllm judge box holds 24 of
# them, so a 16-CPU job there queues behind AssocGrpCpuLimit.

source ~/.bashrc
set -eo pipefail
conda activate qwen25omni
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
mkdir -p results logs

CFG=configs/heard-reply.yaml
ADAPTER=checkpoints/Qwen2.5-Omni-3B-bab-hr2-sft

python -u babble_data.py      --config $CFG
python -u sft_qwen.py         --config $CFG
python -u babble_eval_qwen.py --config $CFG
python -u babble_eval_qwen.py --config $CFG --adapter-path $ADAPTER
