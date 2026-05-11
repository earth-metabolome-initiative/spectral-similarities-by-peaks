#!/bin/bash
# Submit or run the Lawrencium finalization pass.

set -euo pipefail

REPO_DIR="${SPECTRAL_REPO_DIR:-$HOME/spectral-similarities-by-peaks}"
SCRATCH_ROOT="${SPECTRAL_SCRATCH_ROOT:-/global/scratch/users/$USER/spectral-similarities-by-peaks}"
LOGS_DIR="$SCRATCH_ROOT/logs"

usage() {
    cat <<'USAGE'
Usage:
  bash slurm/lrc/finalize.sh <harmonized|gems> [OPTIONS]

Options:
  --partition=PART              SLURM partition (default: lr6)
  --qos=QOS                     SLURM QoS (default: lr_normal)
  --time=HH:MM:SS               Wall time (default: 12:00:00)
  --data-dir=PATH               Dataset cache directory (default: data)
  --output-dir=PATH             Output directory for checkpoints/results
  --neighbors=N                 Top non-self neighbors (default: 64)
  --mz-tolerance=DA             Product m/z tolerance (default: 0.05)
  --row-sample-size=N           GeMS query-row sample size
  --reference-sample-size=N     GeMS reference-column sample size
  --local                       Run finalize-scan immediately instead of sbatch
  --dry-run                     Print command without running/submitting
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
TIME="12:00:00"
DATA_DIR="data"
OUTPUT_DIR=""
NEIGHBORS=64
MZ_TOLERANCE=0.05
ROW_SAMPLE_SIZE=""
REFERENCE_SAMPLE_SIZE=""
LOCAL=false
DRY_RUN=false

case "$PRESET" in
    harmonized)
        OUTPUT_DIR="results/harmonized-full"
        PATHWAY_REPRESENTATIVES=5
        ;;
    gems)
        OUTPUT_DIR="results/gems-sampled"
        PATHWAY_REPRESENTATIVES=0
        ROW_SAMPLE_SIZE=100000
        REFERENCE_SAMPLE_SIZE=1000000
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown preset: $PRESET"
        usage
        exit 1
        ;;
esac

for arg in "$@"; do
    case "$arg" in
        --partition=*)              PARTITION="${arg#*=}" ;;
        --qos=*)                    QOS="${arg#*=}" ;;
        --time=*)                   TIME="${arg#*=}" ;;
        --data-dir=*)               DATA_DIR="${arg#*=}" ;;
        --output-dir=*)             OUTPUT_DIR="${arg#*=}" ;;
        --neighbors=*)              NEIGHBORS="${arg#*=}" ;;
        --mz-tolerance=*)           MZ_TOLERANCE="${arg#*=}" ;;
        --row-sample-size=*)        ROW_SAMPLE_SIZE="${arg#*=}" ;;
        --reference-sample-size=*)  REFERENCE_SAMPLE_SIZE="${arg#*=}" ;;
        --local)                    LOCAL=true ;;
        --dry-run)                  DRY_RUN=true ;;
        -h|--help)                  usage; exit 0 ;;
        *)                          echo "Unknown option: $arg"; usage; exit 1 ;;
    esac
done

mkdir -p "$LOGS_DIR"
cd "$REPO_DIR"

SCAN_ARGS=(
    --dataset "$PRESET"
    --data-dir "$DATA_DIR"
    --output-dir "$OUTPUT_DIR"
    --neighbors "$NEIGHBORS"
    --mz-tolerance "$MZ_TOLERANCE"
)
if [ "$PATHWAY_REPRESENTATIVES" -gt 0 ]; then
    SCAN_ARGS+=(--pathway-representatives-per-class "$PATHWAY_REPRESENTATIVES")
fi
if [ -n "$ROW_SAMPLE_SIZE" ]; then
    SCAN_ARGS+=(--row-sample-size "$ROW_SAMPLE_SIZE")
fi
if [ -n "$REFERENCE_SAMPLE_SIZE" ]; then
    SCAN_ARGS+=(--reference-sample-size "$REFERENCE_SAMPLE_SIZE")
fi

if [ "$LOCAL" = true ]; then
    CMD=(target/release/spectral-similarities-by-peaks finalize-scan)
    CMD+=("${SCAN_ARGS[@]}")
else
    CMD=(
        sbatch
        --partition="$PARTITION"
        --qos="$QOS"
        --time="$TIME"
        --output="$LOGS_DIR/finalize_%j.out"
        --error="$LOGS_DIR/finalize_%j.err"
        slurm/lrc/finalize_job.sh
    )
    CMD+=("${SCAN_ARGS[@]}")
fi

if [ "$DRY_RUN" = true ]; then
    printf '[DRY RUN] '
    printf '%q ' "${CMD[@]}"
    printf '\n'
else
    "${CMD[@]}"
fi
