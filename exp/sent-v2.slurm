#!/bin/bash

#SBATCH --job-name=sent_v2
#SBATCH --account=intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=188G
#SBATCH --gpus=1
#SBATCH --time=1-12:00:00
#SBATCH --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt
#SBATCH --output=logs/sent_v2_%j.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zanqil@uw.edu

# exp/sent-v2.slurm -- re-run a sent track under the current contracts:
# TASK_PROMPT_V2 for SFT and eval, and whichever judge the vLLM box serves
# (resolved inside the python from VLLM_HOST_FILE, so there is nothing to pass
# and nothing to precheck here).
#
#   CFG=configs/sent-4-v2.yaml            sbatch --job-name=sent4v2    exp/sent-v2.slurm
#   CFG=configs/sent-1-asr-v2.yaml BUILD=0 sbatch --job-name=sent1asrv2 exp/sent-v2.slurm
#
# BUILD=0 skips babble_data.py and trains off the dataset the config names --
# that is the sent1asr-v2 case, whose v1 rows are still valid because a
# sent-1-asr probe renders no task prompt (see the config's header). BUILD
# defaults to 1.
#
# Two eval passes, base and adapter, per-kind only -- the C/R/F/EAR table in
# results/viz.ipynb. The type judge + tree matrix (`-typejudge`) is deliberately
# not run: its four-way classification is the older rubric, and every current
# track reads its numbers off the per-kind judges instead.
#
# gpu-l40s/intelligentsystems as exp/bab-sent1asr.slurm: sciencehub/gpu-a40 is
# 52 CPUs total and the vllm judge box holds 24 of them, so a 16-CPU job there
# queues behind AssocGrpCpuLimit.

source ~/.bashrc
set -eo pipefail
conda activate qwen25omni
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
mkdir -p results logs

CFG=${CFG:?set CFG=configs/sent-4-v2.yaml or configs/sent-1-asr-v2.yaml}
BUILD=${BUILD:-1}
# the eval passes name their own output file from the config's `tag`; only the
# adapter path has to be spelled here
ADAPTER=checkpoints/$(python -c "
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['repair_repo_name'])" "$CFG")

# an `[ ... ] && python` one-liner would exit the whole job under `set -e`
# when BUILD=0, since the failed test is the last command of the list
if [ "$BUILD" = 1 ]; then
    python -u babble_data.py --config "$CFG"
fi

python -u sft_qwen.py --config "$CFG"

# the C/R/F/EAR table: one rubric per row kind, scored directly
python -u babble_eval_qwen.py --config "$CFG"
python -u babble_eval_qwen.py --config "$CFG" --adapter-path "$ADAPTER"
