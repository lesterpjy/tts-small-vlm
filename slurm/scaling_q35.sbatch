#!/bin/bash
# Reproduces: Table 2 (tab:rq2), RQ2 scaling -- val-full rows.
# Runs three SC configurations on Qwen3.5-4B, full validation (n=4,651):
#   1. SC-N=8, T=0.7, 1024-tok
#   2. SC-N=8, T=0.7, 2048-tok
#   3. SC-N=16, T=0.7, 2048-tok
#
# Also contributes to Figure 1 (fig:scaling) scaling curves.
#
# Expected wall time:
#   SC-N=8  1k-tok   ~11 h  (8.8 s/q)
#   SC-N=8  2k-tok   ~15 h  (11.8 s/q)
#   SC-N=16 2k-tok   ~19 h  (14.6 s/q)
#   Total            ~45 h  (sequential)
#
# Note: These are long runs. Consider splitting into separate jobs
# if cluster time limits are tight.
#
# Submit:
#   sbatch slurm/scaling_q35.sbatch

#SBATCH --job-name=tts-scaling-q35
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

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

RESUME_FLAG=()
if [[ -n "${RESUME_DIR:-}" ]]; then
    RESUME_FLAG=(--resume-dir "${RESUME_DIR}")
fi

echo "=== RQ2 scaling: Qwen3.5-4B (job $SLURM_JOB_ID) ==="
echo "Resume: ${RESUME_DIR:-<none>}"
nvidia-smi | head -20
echo "===================================="

python scripts/experiment.py \
    --config configs/scaling/q35_sc_n8_1k.yaml \
    "${RESUME_FLAG[@]}" --verbose
echo "[1/3] SC-N=8 1k-tok complete"

python scripts/experiment.py \
    --config configs/scaling/q35_sc_n8_2k.yaml \
    "${RESUME_FLAG[@]}" --verbose
echo "[2/3] SC-N=8 2k-tok complete"

python scripts/experiment.py \
    --config configs/scaling/q35_sc_n16_2k.yaml \
    "${RESUME_FLAG[@]}" --verbose
echo "[3/3] SC-N=16 2k-tok complete"

echo "=== Job $SLURM_JOB_ID finished ==="
