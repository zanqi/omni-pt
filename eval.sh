source ~/.bashrc

JUDGE_HOST=$(cat /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge_host.txt)
JUDGE_URL="http://${JUDGE_HOST}:8000/v1"
curl -sf --max-time 10 "${JUDGE_URL}/models" > /dev/null || { echo "vLLM judge not reachable at ${JUDGE_URL} — start it: sbatch /gscratch/sciencehub/zanqil/vllm_judge/vllm_judge.slurm"; exit 1; } && \
conda activate qwen25omni && \
python babble_eval_qwen.py --model-path Qwen/Qwen2.5-Omni-3B --dataset keylazy/slurp-babble-Qwen2.5-Omni-3B-v4 --judge-base-url "$JUDGE_URL" && \
python babble_eval_qwen.py --model-path Qwen/Qwen2.5-Omni-3B --adapter-path keylazy/Qwen2.5-Omni-3B-bab-sft-adapter --dataset keylazy/slurp-babble-Qwen2.5-Omni-3B-v4 --judge-base-url "$JUDGE_URL" && \
conda activate qwen3omni && \
python babble_eval_qwen.py --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --dataset keylazy/slurp-babble-Qwen3-Omni-30B-A3B-Instruct-v2 --judge-base-url "$JUDGE_URL" && \
python babble_eval_qwen.py --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --adapter-path keylazy/Qwen3-Omni-30B-A3B-Instruct-bab-sft-adapter --dataset keylazy/slurp-babble-Qwen3-Omni-30B-A3B-Instruct-v2 --judge-base-url "$JUDGE_URL"
