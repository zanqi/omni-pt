#!/bin/bash

sbatch --job-name=sent_common --account=sciencehub --partition=gpu-a40 \
  --nodes=1 --cpus-per-task=8 --mem=128G --gpus=1 --time=12:00:00 \
  --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt --output=logs/sent_common_%j.log \
  --mail-type=ALL --mail-user=zanqil@uw.edu \
  --export=all --wrap='./exp/sent-common-run.sh'
