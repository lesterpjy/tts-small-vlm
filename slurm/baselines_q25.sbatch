#!/bin/bash
# Reproduces: Table 1 (tab:rq1), RQ1 baselines -- Qwen2.5-VL-7B rows.
# Runs zero-shot, CoT, and SC-N=8 sequentially on the full EXAMS-V
# validation set (n=4,651), sharing a single model load.
#
# Expected wall time:
#   Zero-shot   ~35 min  (0.63 s/q)
#   CoT         ~5 h     (4.43 s/q)
#   SC-N=8      ~8.5 h   (7.43 s/q)
#   Total       ~14 h
#
# Submit:
#   sbatch slurm/baselines_q25.sbatch

#SBATCH --job-name=tts-baselines-q25
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

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

echo "=== RQ1 baselines: Qwen2.5-VL-7B (job $SLURM_JOB_ID) ==="
nvidia-smi | head -20
python -c "import torch; print(f'torch {torch.__version__} cuda {torch.version.cuda}')"
echo "===================================="

python scripts/experiment.py \
    --config configs/baselines/q25_zero_shot.yaml --verbose
echo "[1/3] Zero-shot complete"

python scripts/experiment.py \
    --config configs/baselines/q25_cot.yaml --verbose
echo "[2/3] CoT complete"

python scripts/experiment.py \
    --config configs/baselines/q25_sc_n8.yaml --verbose
echo "[3/3] SC-N=8 complete"

echo "=== Job $SLURM_JOB_ID finished ==="
