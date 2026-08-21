#!/bin/bash

sbatch --job-name=white_cr --account=sciencehub --partition=gpu-a40 \
  --nodes=1 --cpus-per-task=8 --mem=128G --gpus=1 --time=08:00:00 \
  --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt --output=logs/white_cr_%j.log \
  --mail-type=ALL --mail-user=zanqil@uw.edu \
  --export=all --wrap='./exp/white-cr-run.sh'
