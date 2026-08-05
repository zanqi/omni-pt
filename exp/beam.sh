cd /gscratch/sciencehub/zanqil/projects/omni-pt
DS=keylazy/slurp-babble-Qwen2.5-Omni-3B-beam-v3

DATA=$(BEAM=1 DS_QWEN25=$DS sbatch --parsable babble_data_qwen25.slurm)
echo "data job: $DATA"

BEAM=1 DS_QWEN25=$DS TAG=beam-v3 sbatch \
    --dependency=afterok:$DATA --kill-on-invalid-dep=yes sft_eval_qwen25.slurm
