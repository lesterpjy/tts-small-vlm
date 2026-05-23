#!/usr/bin/env bash
# Launch vLLM server for batch inference.
# The primary pipeline uses HuggingFace Transformers directly.
# This is only needed if you switch to the vLLM backend.
#
# Usage:
#   ./scripts/serve_vllm.sh                  # default: FP16 mode
#   ./scripts/serve_vllm.sh --quantized      # INT4 quantized mode

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${PORT:-8000}"

if [[ "${1:-}" == "--quantized" ]]; then
    echo "Quantized mode: INT4, 32k context"
    vllm serve "$MODEL" \
        --dtype bfloat16 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.9 \
        --quantization awq_marlin \
        --limit-mm-per-prompt image=1 \
        --port "$PORT"
else
    echo "Default mode: FP16, 64k context"
    vllm serve "$MODEL" \
        --dtype bfloat16 \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.85 \
        --limit-mm-per-prompt image=1 \
        --port "$PORT"
fi
