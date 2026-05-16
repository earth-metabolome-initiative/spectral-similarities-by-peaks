#!/bin/bash
# Submit the AUROC / AUPRC computation for an existing scan output.
#
# The streaming implementation walks `pathway_shards/<config>/top_<k>/` and
# processes one shard at a time, so it can run on a single Lawrencium node
# even when the merged `pathway_scores.parquet` would be too large.

set -euo pipefail

REPO_DIR="${SPECTRAL_REPO_DIR:-$HOME/spectral-similarities-by-peaks}"
SCRATCH_ROOT="${SPECTRAL_SCRATCH_ROOT:-/global/scratch/users/$USER/spectral-similarities-by-peaks}"
LOGS_DIR="$SCRATCH_ROOT/logs"

usage() {
    cat <<'USAGE'
Usage:
  bash slurm/lrc/compute_pathway_discriminability.sh harmonized [OPTIONS]

Only the `harmonized` preset is supported because pathway classification
requires NPC pathway annotations, which the GeMS-A10 corpus does not have.

Options:
  --partition=PART      SLURM partition (default: lr6)
  --qos=QOS             SLURM QoS       (default: lr_normal)
  --time=HH:MM:SS       Wall time       (default: 06:00:00)
  --output-dir=PATH     Override the default per-preset output directory
  --local               Run the command directly instead of sbatch
  --dry-run             Print the sbatch invocation without submitting
USAGE
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

PRESET="$1"
shift

PARTITION="lr6"
QOS="lr_normal"
TIME="06:00:00"
OUTPUT_DIR=""
LOCAL=false
DRY_RUN=false

case "$PRESET" in
    harmonized) OUTPUT_DIR="results/harmonized-full" ;;
    -h|--help)  usage; exit 0 ;;
    *)          echo "Unknown preset: $PRESET"; usage; exit 1 ;;
esac

for arg in "$@"; do
    case "$arg" in
        --partition=*)  PARTITION="${arg#*=}"  ;;
        --qos=*)        QOS="${arg#*=}"        ;;
        --time=*)       TIME="${arg#*=}"       ;;
        --output-dir=*) OUTPUT_DIR="${arg#*=}" ;;
        --local)        LOCAL=true             ;;
        --dry-run)      DRY_RUN=true           ;;
        -h|--help)      usage; exit 0          ;;
        *)              echo "Unknown option: $arg"; usage; exit 1 ;;
    esac
done

mkdir -p "$LOGS_DIR"
cd "$REPO_DIR"

JOB_NAME="spectral-$PRESET-pathway-discrim"

if [ "$LOCAL" = true ]; then
    CMD=(target/release/spectral-similarities-by-peaks compute-pathway-discriminability --output-dir "$OUTPUT_DIR")
else
    CMD=(
        sbatch
        --partition="$PARTITION"
        --qos="$QOS"
        --time="$TIME"
        --job-name="$JOB_NAME"
        --output="$LOGS_DIR/pathway_discrim_${PRESET}_%j.out"
        --error="$LOGS_DIR/pathway_discrim_${PRESET}_%j.err"
        slurm/lrc/compute_pathway_discriminability_job.sh
        --output-dir "$OUTPUT_DIR"
    )
fi

if [ "$DRY_RUN" = true ]; then
    printf '[DRY RUN] '
    printf '%q ' "${CMD[@]}"
    printf '\n'
else
    "${CMD[@]}"
fi
