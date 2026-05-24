#!/bin/bash
# Reproduces: Table 1 (tab:rq1), RQ1 baselines -- Qwen3.5-4B rows.
# Runs zero-shot and CoT on the full EXAMS-V validation set (n=4,651).
#
# Expected wall time:
#   Zero-shot   ~1 h   (0.63 s/q)
#   CoT         ~6 h   (4.43 s/q)
#   Total       ~7 h
#
# Submit:
#   sbatch slurm/baselines_q35.sbatch

#SBATCH --job-name=tts-baselines-q35
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

echo "=== RQ1 baselines: Qwen3.5-4B (job $SLURM_JOB_ID) ==="
nvidia-smi | head -20
python -c "import torch; print(f'torch {torch.__version__} cuda {torch.version.cuda}')"
echo "===================================="

python scripts/experiment.py \
    --config configs/baselines/q35_zero_shot.yaml --verbose
echo "[1/2] Zero-shot complete"

python scripts/experiment.py \
    --config configs/baselines/q35_cot.yaml --verbose
echo "[2/2] CoT complete"

echo "=== Job $SLURM_JOB_ID finished ==="
