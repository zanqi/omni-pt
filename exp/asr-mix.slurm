#!/bin/bash

#SBATCH --job-name=asr_mix
#SBATCH --account=intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=188G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt
#SBATCH --output=logs/asr_mix_%j.log
#SBATCH --export=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zanqil@uw.edu

# exp/asr-mix.slurm -- one arm of steps/asr-mix.html, chosen by CFG.
#
#   CFG=configs/asr1.yaml sbatch --job-name=asr1 exp/asr-mix.slurm
#   CFG=configs/asr4.yaml sbatch --job-name=asr4 exp/asr-mix.slurm
#
# asr4's build is ~26 h and asr1's ~16 h; both fit the 2-day wall with the SFT
# and the four eval passes after them. To redo one stage, comment the others
# out and resubmit -- every stage is idempotent except the build, which
# rmtree's its own audio dir.

source ~/.bashrc
set -eo pipefail
conda activate qwen25omni
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
mkdir -p results logs

CFG=${CFG:?set CFG=configs/asr1.yaml or configs/asr4.yaml}
TAG=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['tag'])" "$CFG")
ADAPTER=checkpoints/Qwen2.5-Omni-3B-bab-${TAG}-sft

python -u babble_data.py      --config "$CFG"
python -u sft_qwen.py         --config "$CFG"

# the premise check: does the mixed adapter hear like the labeler it trusted?
python -u asr_eval_qwen.py --config "$CFG" --tag "${TAG}-mixed" --adapter-path "$ADAPTER"

python -u babble_eval_qwen.py --config "$CFG"
python -u babble_eval_qwen.py --config "$CFG" --adapter-path "$ADAPTER"
