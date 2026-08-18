#!/bin/bash

# sent-loss over the tree track's two witnesses (ASR transcript + task reply),
# labeled in two stages: one call inventories the command's key pieces, then one
# call per witness returns the ids it lost. Scored twice off the one adapter --
# per-kind (comparable with beam) and type judge + tree matrix (comparable with
# tree-v3) -- so the labeling change can be read under either rubric.

sbatch --job-name=sent2_v1_e2e --account=cse --partition=gpu-l40s \
  --nodes=1 --cpus-per-task=16 --mem=188G --gpus=1 --time=1-00:00:00 \
  --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt --output=logs/sent2_v1_e2e_%j.log \
  --mail-type=ALL --mail-user=zanqil@uw.edu \
  --export=all --wrap='DS=keylazy/slurp-babble-Qwen2.5-Omni-3B-sent2-v1; \
    SENT2=1 DS_QWEN25=$DS ./babble_data.sh qwen25 && \
    SENT2=1 DS_QWEN25=$DS MULTS= ./sft.sh qwen25 && \
    SENT2=1 DS_QWEN25=$DS MULTS= TAG=sent2-v1 ./eval.sh qwen25 && \
    SENT2=1 DS_QWEN25=$DS MULTS= TAG=sent2-v1-typejudge \
      EVAL_FLAGS="--score-matrix tree --restate-prompt" ./eval.sh qwen25'


# eval only
# SENT2=1 MULTS= TAG=sent2-v1 DS_QWEN25=keylazy/slurp-babble-Qwen2.5-Omni-3B-sent2-v1 ./eval.sh qwen25 2>&1 | tee logs/sent2_v1_eval.log
