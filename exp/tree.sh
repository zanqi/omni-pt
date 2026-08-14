#!/bin/bash

sbatch --job-name=tree_v3_e2e --account=intelligentsystems --partition=gpu-l40s \
  --nodes=1 --cpus-per-task=16 --mem=188G --gpus=1 --time=2-00:00:00 \
  --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt --output=logs/tree_v3_e2e_%j.log \
  --mail-type=ALL --mail-user=zanqil@uw.edu \
  --export=all --wrap='DS=keylazy/slurp-babble-Qwen2.5-Omni-3B-tree-v3; \
    TREE=1 DS_QWEN25=$DS ./babble_data.sh qwen25 && \
    TREE=1 DS_QWEN25=$DS MULTS= ./sft.sh qwen25 && \
    TREE=1 DS_QWEN25=$DS MULTS= TAG=tree-v3 ./eval.sh qwen25'


# eval only
# TREE=1 MULTS= TAG=tree-v2 DS_QWEN25=keylazy/slurp-babble-Qwen2.5-Omni-3B-tree-v2 ./eval.sh qwen25 2>&1 | tee logs/tree_v2_eval.log