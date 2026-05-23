#!/bin/bash
# Reproduces: Appendix Table (tab:rq4b) -- PRM scoring mode comparison.
# Two-step workflow on Q2.5 dev-200:
#   Step 1: Generate DTR N=4,M=4 candidates (Qwen2.5-VL-7B)
#   Step 2a: PRM segmented (per-step) rescoring
#   Step 2b: PRM flat (one-shot) rescoring
#
# Both policy and PRM co-reside on one GPU (~34 GB peak).
# Policy generates first, then PRM rescores from JSONL.
#
# Expected wall time:
#   Step 1  (DTR N=4,M=4, 200 questions): ~1.5 h
#   Step 2a (PRM segmented):              ~4.6 h
#   Step 2b (PRM flat):                   ~0.5 h
#   Total:                                ~6.6 h
#
# Resume: set RESUME=1 and DTR_RUN_DIR to skip Step 1.
#
# Submit:
#   sbatch slurm/dtr_prm_rescore_dev.sbatch
#   sbatch --export=RESUME=1,DTR_RUN_DIR=runs/dtr_... slurm/dtr_prm_rescore_dev.sbatch

#SBATCH --job-name=tts-dtr-prm-dev
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

echo "=== DTR + PRM rescore: Q2.5 dev-200 (job $SLURM_JOB_ID) ==="
nvidia-smi | head -20
python -c "import torch; print(f'torch {torch.__version__} cuda {torch.version.cuda}')"
echo "===================================="

# --- Step 1: Generate DTR candidates ---
RESUME="${RESUME:-0}"

if [ "$RESUME" = "1" ] && [ -n "${DTR_RUN_DIR:-}" ]; then
    DTR_RUN="$DTR_RUN_DIR"
    echo ""
    echo "=== RESUME MODE: skipping Step 1, using existing $DTR_RUN ==="
    if [ ! -f "$DTR_RUN/candidates.jsonl" ]; then
        echo "[ERR] $DTR_RUN/candidates.jsonl not found"
        exit 1
    fi
    wc -l "$DTR_RUN/candidates.jsonl"
else
    echo ""
    echo "=== Step 1: DTR N=4,M=4 on dev-200 ==="
    python scripts/experiment.py \
        --config configs/search/q25_dtr_n4m4_prm.yaml \
        --verbose

    # Find the generated run directory
    DTR_RUN=$(ls -dt runs/*dtr*n4m4*prm* 2>/dev/null | head -1)
    if [ -z "$DTR_RUN" ]; then
        echo "[ERR] No DTR run directory found; step 1 may have failed."
        exit 1
    fi
    echo "DTR candidates: $DTR_RUN/candidates.jsonl"
    wc -l "$DTR_RUN/candidates.jsonl"
fi

# --- Step 2a: PRM segmented rescoring ---
RESUME_FLAG=""
if [ "$RESUME" = "1" ]; then
    RESUME_FLAG="--resume"
fi

echo ""
echo "=== Step 2a: PRM segmented rescoring ==="
python scripts/rescore_dtr_with_prm.py \
    --input  "$DTR_RUN/candidates.jsonl" \
    --output "$DTR_RUN/prm_segmented_scored.jsonl" \
    --dataset examsv_validation \
    --mode segmented \
    --aggregation overall \
    $RESUME_FLAG

# --- Step 2b: PRM flat rescoring ---
echo ""
echo "=== Step 2b: PRM flat (one-shot) rescoring ==="
python scripts/rescore_dtr_with_prm.py \
    --input  "$DTR_RUN/candidates.jsonl" \
    --output "$DTR_RUN/prm_flat_scored.jsonl" \
    --dataset examsv_validation \
    --mode flat_one_shot \
    $RESUME_FLAG

echo ""
echo "=== Job $SLURM_JOB_ID finished ==="
echo "Outputs:"
echo "  DTR candidates:  $DTR_RUN/candidates.jsonl"
echo "  PRM segmented:   $DTR_RUN/prm_segmented_scored.jsonl"
echo "  PRM flat:        $DTR_RUN/prm_flat_scored.jsonl"
