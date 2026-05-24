#!/bin/bash
# Reproduces: Table 4 (tab:rq4), RQ4 selectors -- Qwen3.5-4B rows.
# Rescores the Q3.5 SC-N=8 2k-tok candidate pool (val-full) with:
#   1. R1 generative critic (GM-PRM-style, ~3 extra calls/chain)
#   2. R2 Qwen-VL-PRM one-shot scoring (~1 call/chain)
#
# Requires: a completed SC-N=8 2k-tok Q3.5 val-full run with candidates.jsonl.
# Set POOL_DIR to the run directory, or the script auto-detects it.
#
# Expected wall time:
#   R1 critic (with --skip-high-confidence): ~5-6 h
#   R2 PRM one-shot (with --skip-high-confidence): ~2 h
#   Total: ~8 h
#
# Submit:
#   sbatch slurm/selectors_q35.sbatch
# Or with explicit pool:
#   sbatch --export=ALL,POOL_DIR=runs/sc_n8_q35_2k_val-... slurm/selectors_q35.sbatch

#SBATCH --job-name=tts-selectors-q35
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

# Adapt module loads for your HPC system (versions below are for SURFsara Snellius).
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
# shellcheck source=/dev/null
source .venv/bin/activate

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"

mkdir -p logs runs

# Auto-detect pool directory if not set
POOL_DIR="${POOL_DIR:-}"
if [[ -z "$POOL_DIR" ]]; then
    POOL_DIR=$(ls -1d runs/*sc_n8*q35*2k*val* 2>/dev/null | tail -1)
fi
if [[ -z "$POOL_DIR" || ! -f "$POOL_DIR/candidates.jsonl" ]]; then
    echo "ERROR: no SC-N=8 2k Q3.5 val pool found. Set POOL_DIR or run scaling_q35 first." >&2
    exit 1
fi

echo "=== RQ4 selectors: Qwen3.5 pool (job $SLURM_JOB_ID) ==="
nvidia-smi | head -20
echo "Pool: $POOL_DIR"
echo "===================================="

# R1: generative critic
echo ""
echo "=== R1: Generative critic ==="
python scripts/score_with_critic.py \
    --input  "$POOL_DIR/candidates.jsonl" \
    --output "$POOL_DIR/critic_scored.jsonl" \
    --dataset examsv_validation \
    --model Qwen/Qwen3.5-4B \
    --backend vllm \
    --max-model-len 16384 \
    --skip-high-confidence

# R2: PRM one-shot (cross-policy: PRM trained on Q2.5, scoring Q3.5 chains)
echo ""
echo "=== R2: PRM one-shot rescore (cross-policy) ==="
python scripts/score_with_prm.py \
    --input  "$POOL_DIR/candidates.jsonl" \
    --output "$POOL_DIR/prm_scored.jsonl" \
    --dataset examsv_validation \
    --mode one_shot \
    --skip-high-confidence

echo "=== Job $SLURM_JOB_ID finished ==="
echo "Outputs:"
echo "  Critic scored: $POOL_DIR/critic_scored.jsonl"
echo "  PRM scored:    $POOL_DIR/prm_scored.jsonl"
