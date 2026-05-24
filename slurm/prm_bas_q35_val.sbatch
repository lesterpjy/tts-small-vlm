#!/bin/bash
# Reproduces: Section 5.1 (sec:dtr-bas) PRM-BAS result for Qwen3.5-4B.
# Runs PRM-BAS beam-annealing search on Qwen3.5-4B over the full
# validation set (n=4,651). The paper reports 80.74% on n=4,319
# completed questions vs SC majority at 81.13%.
#
# PRM-BAS is sequential per-beam. Both policy and PRM co-reside
# on a single GPU (~34 GB peak).
#
# This is a long job. Uses sharded execution for resilience:
# set SHARDS_TO_RUN to control which slices to process.
#
# Expected wall time per shard (~582 questions): ~16 h at ~100 s/q.
# Single full run: ~130 h. Use multiple shards across jobs.
#
# Submit (example with 2 parallel batches):
#   sbatch --export=ALL,SHARDS_TO_RUN="0 1 2 3" slurm/prm_bas_q35_val.sbatch
#   sbatch --export=ALL,SHARDS_TO_RUN="4 5 6 7" slurm/prm_bas_q35_val.sbatch

#SBATCH --job-name=tts-prm-bas-q35-val
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -uo pipefail  # NOT -e: a single shard failure should not kill others

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

SHARDS_TO_RUN="${SHARDS_TO_RUN:-}"
if [[ -z "$SHARDS_TO_RUN" ]]; then
    echo "ERROR: set SHARDS_TO_RUN, e.g. --export=ALL,SHARDS_TO_RUN='0 1 2 3'" >&2
    exit 2
fi

TOTAL=4651
NUM_SHARDS=8
PER_SHARD=$(( (TOTAL + NUM_SHARDS - 1) / NUM_SHARDS ))  # ceil = 582

echo "=== PRM-BAS: Qwen3.5-4B val-full sharded (job $SLURM_JOB_ID) ==="
echo "Shards: ${SHARDS_TO_RUN}"
echo "Per-shard slice size: ${PER_SHARD}"
nvidia-smi | head -20
echo "===================================="

declare -a SHARD_RC
for shard in $SHARDS_TO_RUN; do
    START=$(( shard * PER_SHARD ))
    END=$(( START + PER_SHARD ))

    # Look for existing run directory to resume from
    RESUME_DIR=$(ls -d runs/*prm_bas*q35*val*shard${shard}-* 2>/dev/null | tail -1)
    RESUME_FLAG=()
    if [[ -n "$RESUME_DIR" ]]; then
        RESUME_FLAG=(--resume-dir "${RESUME_DIR}")
    fi

    echo ""
    echo "================================================================"
    echo "[shard ${shard}] slice=[${START}:${END}]  resume=${RESUME_DIR:-<fresh>}"
    echo "[shard ${shard}] started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "================================================================"

    python scripts/experiment.py \
        --config configs/search/q35_prm_bas_val.yaml \
        --start-idx "${START}" \
        --end-idx "${END}" \
        --variant-suffix "shard${shard}" \
        "${RESUME_FLAG[@]}" \
        --verbose
    rc=$?

    echo "[shard ${shard}] finished $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc}"
    SHARD_RC+=("${shard}:rc=${rc}")
done

echo ""
echo "================================================================"
echo "=== Batch summary (job $SLURM_JOB_ID) ==="
for entry in "${SHARD_RC[@]}"; do echo "  $entry"; done
echo "================================================================"
echo "=== Job $SLURM_JOB_ID finished ==="
