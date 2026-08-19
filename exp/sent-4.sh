#!/bin/bash

# sent-loss over the beam track's four hypotheses: same N-best ASR pass and no
# task-response pass, but each hypothesis is scored against one key-piece
# inventory and a piece counts lost only if all four missed it. Scored twice off
# the one adapter -- per-kind (comparable with beam-v3) and type judge + tree
# matrix -- with no restate prompt on either pass, since these rows were probed
# and trained under the plain TASK_PROMPT.

sbatch --job-name=sent4_v1_e2e --account=sciencehub --partition=gpu-a40 \
  --nodes=1 --cpus-per-task=8 --mem=128G --gpus=1 --time=1-00:00:00 \
  --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt --output=logs/sent4_v1_e2e_%j.log \
  --mail-type=ALL --mail-user=zanqil@uw.edu \
  --export=all --wrap='DS=keylazy/slurp-babble-Qwen2.5-Omni-3B-sent4-v1; \
    SENT4=1 DS_QWEN25=$DS ./babble_data.sh qwen25 && \
    SENT4=1 DS_QWEN25=$DS MULTS= ./sft.sh qwen25 && \
    SENT4=1 DS_QWEN25=$DS MULTS= TAG=sent4-v1 ./eval.sh qwen25 && \
    SENT4=1 DS_QWEN25=$DS MULTS= TAG=sent4-v1-typejudge \
      EVAL_FLAGS="--score-matrix tree" ./eval.sh qwen25'


# eval only
# SENT4=1 MULTS= TAG=sent4-v1 DS_QWEN25=keylazy/slurp-babble-Qwen2.5-Omni-3B-sent4-v1 ./eval.sh qwen25 2>&1 | tee logs/sent4_v1_eval.log
