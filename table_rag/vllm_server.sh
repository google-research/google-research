MODEL=$1

echo "Starting VLLM API server with model $MODEL"

vllm serve $MODEL --dtype auto --api-key token-abc123 --max_model_len 64000