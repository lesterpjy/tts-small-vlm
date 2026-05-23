#!/bin/bash
# Reproduces: Table 3 (tab:rq1b), search and verification on dev-200.
# Runs four search strategies under Qwen2.5-VL-7B on the 200-question
# stratified development subset:
#   1. SC-N=8 (baseline anchor)
#   2. DTR N=2, M=2
#   3. DTR N=4, M=4
#   4. PRM-BAS (DTR N=2 + beam-annealing with Qwen-VL-PRM)
#
# Expected wall time:
#   SC-N=8    ~25 min  (7.4 s/q * 200)
#   DTR N2M2  ~47 min  (14.1 s/q * 200)
#   DTR N4M4  ~1.6 h   (28.5 s/q * 200)
#   PRM-BAS   ~2.4 h   (42.3 s/q * 200)
#   Total     ~5 h
#
# Submit:
#   sbatch slurm/search_q25_dev.sbatch

#SBATCH --job-name=tts-search-q25-dev
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=08:00:00
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

echo "=== Search strategies: Qwen2.5 dev-200 (job $SLURM_JOB_ID) ==="
nvidia-smi | head -20
python -c "import torch; print(f'torch {torch.__version__} cuda {torch.version.cuda}')"
echo "===================================="

python scripts/experiment.py \
    --config configs/search/q25_sc_n8_dev.yaml --verbose
echo "[1/4] SC-N=8 complete"

python scripts/experiment.py \
    --config configs/search/q25_dtr_n2m2.yaml --verbose
echo "[2/4] DTR N=2,M=2 complete"

python scripts/experiment.py \
    --config configs/search/q25_dtr_n4m4.yaml --verbose
echo "[3/4] DTR N=4,M=4 complete"

python scripts/experiment.py \
    --config configs/search/q25_prm_bas_dev.yaml --verbose
echo "[4/4] PRM-BAS complete"

echo "=== Job $SLURM_JOB_ID finished ==="
