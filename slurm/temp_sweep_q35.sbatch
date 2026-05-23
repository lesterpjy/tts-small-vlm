#!/bin/bash
# Reproduces: Table 2 (tab:rq2), temperature sweep rows.
# Runs SC-N=8 at 2048-tok on Qwen3.5-4B with T in {0.3, 0.5, 0.7, 0.9}
# on the full validation set (n=4,651).
#
# Note: T=0.7 is the same configuration as SC-N=8 2k in scaling_q35.
# If that run already exists, you can skip T=0.7 here.
#
# Expected wall time per config: ~15 h (11.8 s/q * 4651).
# Running all 4 sequentially: ~60 h. Consider splitting if needed.
#
# Submit:
#   sbatch slurm/temp_sweep_q35.sbatch

#SBATCH --job-name=tts-temp-sweep-q35
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

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

RESUME_FLAG=()
if [[ -n "${RESUME_DIR:-}" ]]; then
    RESUME_FLAG=(--resume-dir "${RESUME_DIR}")
fi

echo "=== Temperature sweep: Qwen3.5-4B SC-N=8 2k (job $SLURM_JOB_ID) ==="
echo "Resume: ${RESUME_DIR:-<none>}"
nvidia-smi | head -20
echo "===================================="

python scripts/experiment.py \
    --config configs/scaling/q35_sc_n8_2k_t03.yaml \
    "${RESUME_FLAG[@]}" --verbose
echo "[1/4] T=0.3 complete"

python scripts/experiment.py \
    --config configs/scaling/q35_sc_n8_2k_t05.yaml \
    "${RESUME_FLAG[@]}" --verbose
echo "[2/4] T=0.5 complete"

# T=0.7 is the default config (same as scaling/q35_sc_n8_2k.yaml).
# Skip if already run via scaling_q35.sbatch.
python scripts/experiment.py \
    --config configs/scaling/q35_sc_n8_2k.yaml \
    "${RESUME_FLAG[@]}" --verbose
echo "[3/4] T=0.7 complete"

python scripts/experiment.py \
    --config configs/scaling/q35_sc_n8_2k_t09.yaml \
    "${RESUME_FLAG[@]}" --verbose
echo "[4/4] T=0.9 complete"

echo "=== Job $SLURM_JOB_ID finished ==="
