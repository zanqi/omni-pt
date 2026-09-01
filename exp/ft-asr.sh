#!/bin/bash

# exp/ft-asr.sh -- submitter for both phases of the ft-asr track
# (steps/ft-asr.html, steps/ft-asr-2.html).
#
#   ./exp/ft-asr.sh data    # stage 0: ASR dataset, ASR LoRA, the diagnostic.
#                           # STOP here and read results/asr_*_{base,ft}.jsonl
#                           # against steps/ft-asr-2.html step 8.
#   ./exp/ft-asr.sh sent4   # stages 1-3: sent-4 build off the adapter, SFT,
#                           # and both judge passes.
#
# Split at the gate on purpose: the second phase is ~a day of GPU and is only
# worth submitting if the first phase's WER and distinct-hypothesis numbers
# clear the thresholds.
set -eo pipefail
PHASE="${1:?usage: ./exp/ft-asr.sh data|sent4}"

sbatch --job-name=ftasr_${PHASE} --account=sciencehub --partition=gpu-a40 \
  --nodes=1 --cpus-per-task=8 --mem=128G --gpus=1 --time=1-00:00:00 \
  --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt \
  --output=logs/ftasr_${PHASE}_%j.log \
  --mail-type=ALL --mail-user=zanqil@uw.edu \
  --export=all,PHASE=${PHASE} --wrap='./exp/ft-asr-run.sh'
