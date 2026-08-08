#!/bin/bash

sbatch --job-name=beam --account=intelligentsystems --partition=gpu-l40s \
  --nodes=1 --cpus-per-task=16 --mem=188G --gpus=1 --time=2-00:00:00 \
  --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt --output=logs/beam_v3_e2e_%j.log \
  --mail-type=ALL --mail-user=zanqil@uw.edu \
  --export=all --wrap='DS=keylazy/slurp-babble-Qwen2.5-Omni-3B-beam-v3; \
    BEAM=1 DS_QWEN25=$DS ./babble_data.sh qwen25 && \
    BEAM=1 DS_QWEN25=$DS MULTS= ./sft.sh qwen25 && \
    BEAM=1 DS_QWEN25=$DS MULTS= TAG=beam-v3 ./eval.sh qwen25'


# eval only
# BEAM=1 MULTS= TAG=beam-v3 DS_QWEN25=keylazy/slurp-babble-Qwen2.5-Omni-3B-beam-v3 ./eval.sh qwen25 2>&1 | tee logs/beam_v3_eval.log
