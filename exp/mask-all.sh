#!/bin/bash

#SBATCH --job-name=mask_all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zanqil@uw.edu

#SBATCH --account=sciencehub
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=188G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00

#SBATCH --chdir=/gscratch/sciencehub/zanqil/projects/omni-pt
#SBATCH --output=logs/mask_all_%j.log
#SBATCH --export=all

# exp/mask-all.sh -- the whole mask track end to end in one job.
#
#   sbatch exp/mask-all.sh                            # build, SFT, DPO, eval
#   STAGES="sft dpo eval" sbatch exp/mask-all.sh      # reuse an existing dataset
#   N_TRAIN=200 N_TEST=20 sbatch exp/mask-all.sh      # a short shakedown run
#   ./exp/mask-all.sh                                 # or run it in a shell you hold
#
# One job rather than four: the stages are strictly sequential (each trains on
# what the last one produced), so splitting them only adds three queue waits,
# and the 16 CPUs the build wants are the binding request. The four stages are
# the four exp/ scripts that already exist, run in order as child processes --
# each is untouched and still `sbatch exp/<stage>.sh`-able on its own, and each
# activates its own conda env: mask_data.py needs qwen3omni for the forced
# aligner, everything after it needs qwen25omni.
#
# --time is 2 days for the whole track. The v1 stages were budgeted 12h + 12h +
# 16h + 12h separately; those were queue-safety ceilings, not measurements, and
# a job that only holds the GPU as long as it needs costs nothing extra.
#
# The v2 defaults are the three-kind build: repeat rows alongside answer and
# repair, so the eval reports F and EAR = 3*C*R*F/(C*R + C*F + R*F).

source ~/.bashrc
set -eo pipefail

DS_ID="${DS_ID:-keylazy/slurp-mask-v2}"
N_TRAIN="${N_TRAIN:-1000}"   # utterances -> x3 kinds = 3000 train rows
N_TEST="${N_TEST:-100}"      # utterances -> x3 kinds x4 masks = 1200 test rows
SFT_RUN="${SFT_RUN:-Qwen2.5-Omni-3B-mask-v2-sft}"
DPO_RUN="${DPO_RUN:-Qwen2.5-Omni-3B-mask-v2-dpo}"
TAG="${TAG:-mask-v2}"
STAGES="${STAGES:-data sft dpo eval}"

SFT_ADAPTER="checkpoints/${SFT_RUN}"
DPO_ADAPTER="checkpoints/${DPO_RUN}"

mkdir -p results logs

# fail-fast before an hour of building: three of the four stages call the vLLM
# box (targets, DPO ranking, judging), and the two training stages are the only
# ones that do not
JUDGE_HOST=$(cat /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge_host.txt)
if ! curl -sf --max-time 10 "http://${JUDGE_HOST}:8000/models" > /dev/null \
   && ! curl -sf --max-time 10 "http://${JUDGE_HOST}:8000/v1/models" > /dev/null; then
    echo "vLLM box not reachable at ${JUDGE_HOST}:8000 -- sbatch /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge.slurm" >&2
    exit 1
fi
echo "=== mask track: ${DS_ID} | stages: ${STAGES} | vLLM ${JUDGE_HOST} ==="

for stage in $STAGES; do
    echo ""
    echo "############ ${stage} ############"
    case "$stage" in
        data)
            DS_ID="$DS_ID" N_TRAIN="$N_TRAIN" N_TEST="$N_TEST" \
                bash exp/mask_data.sh ;;
        sft)
            DS_ID="$DS_ID" RUN_NAME="$SFT_RUN" bash exp/mask-sft.sh ;;
        dpo)
            DS_ID="$DS_ID" SFT_ADAPTER="$SFT_ADAPTER" RUN_NAME="$DPO_RUN" \
                bash exp/mask-dpo.sh ;;
        eval)
            # one test split, one judge, three lines -- base is what makes the
            # SFT and DPO numbers mean anything
            DS_ID="$DS_ID" TAG="$TAG" STAGES="base sft dpo" \
                SFT_ADAPTER="$SFT_ADAPTER" DPO_ADAPTER="$DPO_ADAPTER" \
                bash exp/mask.sh ;;
        *)
            echo "!! unknown stage ${stage}" >&2; exit 1 ;;
    esac
done

echo ""
echo "=== done: results/mask_results_*_${TAG}.jsonl ==="

# TODO: it haven't finished. continue by:
# STAGES="dpo eval" sbatch exp/mask-all.sh

