#!/bin/bash
# Reproduces: Best configuration (S-best in Table 1, tab:arms).
# Two-step job for competition submission on EXAMS-V test (n=1,117):
#   Step 1: SC-N=16 with 2k-tok on Qwen3.5-4B (best val config: 81.64%)
#   Step 2: Guided repair of parse-fail questions
#
# Expected wall time:
#   Step 1: 1117 * 16 chains * ~7 s/chain ~ 35 h
#   Step 2: ~10 min (parse-fail repair on ~80 questions)
#   Total:  ~36 h
#
# Submit:
#   sbatch slurm/test_submission.sbatch

#SBATCH --job-name=tts-test-submit
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=48:00:00
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

mkdir -p logs runs submissions

RESUME_FLAG=()
if [[ -n "${RESUME_DIR:-}" ]]; then
    RESUME_FLAG=(--resume-dir "${RESUME_DIR}")
fi

echo "=== Test submission: Qwen3.5 SC-N=16 2k (job $SLURM_JOB_ID) ==="
nvidia-smi | head -20
echo "===================================="

# Step 1: Generate SC-N=16 candidates on test split
python scripts/experiment.py \
    --config configs/test/q35_sc_n16_2k_test.yaml \
    "${RESUME_FLAG[@]}" \
    --verbose

# Find the run directory
RUN_DIR=$(ls -dt runs/*sc_n16*2k*test* 2>/dev/null | head -1)
if [[ -z "$RUN_DIR" ]]; then
    echo "[ERR] No test run directory found."
    exit 1
fi
echo "Run directory: $RUN_DIR"

# Step 2: Guided repair of parse-fail questions
echo ""
echo "=== Guided repair of parse failures ==="
python scripts/repair_parse_failures.py \
    --candidates "$RUN_DIR/candidates.jsonl" \
    --dataset imageclef_mcq_test \
    --model "Qwen/Qwen3.5-4B"

# Step 3: Format competition submission
echo ""
echo "=== Formatting submission ==="
python scripts/format_submission.py \
    --input "$RUN_DIR/candidates.jsonl" \
    --output submissions/test_submission.json \
    --expected-count 1117

echo "=== Job $SLURM_JOB_ID finished ==="
echo "Submission: submissions/test_submission.json"
