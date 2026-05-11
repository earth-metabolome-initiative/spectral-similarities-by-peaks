#!/bin/bash
# Submit or run the Lawrencium dataset prefetch pass.

set -euo pipefail

REPO_DIR="${SPECTRAL_REPO_DIR:-$HOME/spectral-similarities-by-peaks}"
SCRATCH_ROOT="${SPECTRAL_SCRATCH_ROOT:-/global/scratch/users/$USER/spectral-similarities-by-peaks}"
LOGS_DIR="$SCRATCH_ROOT/logs"

usage() {
    cat <<'USAGE'
Usage:
  bash slurm/lrc/prefetch.sh <harmonized|gems> [OPTIONS]

Options:
  --partition=PART        SLURM partition (default: lr6)
  --qos=QOS               SLURM QoS (default: lr_normal)
  --time=HH:MM:SS         Wall time (default: 06:00:00)
  --data-dir=PATH         Dataset cache directory (default: data)
  --gems-parts=N[,N...]   GeMS part numbers to load
  --no-wait               Submit prefetch without waiting for completion
  --local                 Run prefetch immediately instead of sbatch
  --dry-run               Print command without running/submitting
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
DATA_DIR="data"
GEMS_PARTS=""
WAIT=true
LOCAL=false
DRY_RUN=false

case "$PRESET" in
    harmonized|gems)
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
        --partition=*)     PARTITION="${arg#*=}" ;;
        --qos=*)           QOS="${arg#*=}" ;;
        --time=*)          TIME="${arg#*=}" ;;
        --data-dir=*)      DATA_DIR="${arg#*=}" ;;
        --gems-parts=*)    GEMS_PARTS="${arg#*=}" ;;
        --no-wait)         WAIT=false ;;
        --local)           LOCAL=true ;;
        --dry-run)         DRY_RUN=true ;;
        -h|--help)         usage; exit 0 ;;
        *)                 echo "Unknown option: $arg"; usage; exit 1 ;;
    esac
done

mkdir -p "$LOGS_DIR"
cd "$REPO_DIR"

PREFETCH_ARGS=(
    --dataset "$PRESET"
    --data-dir "$DATA_DIR"
)
if [ -n "$GEMS_PARTS" ]; then
    PREFETCH_ARGS+=(--gems-parts "$GEMS_PARTS")
fi

PREFETCH_JOB_NAME="spectral-$PRESET-prefetch"

if [ "$LOCAL" = true ]; then
    CMD=(target/release/spectral-similarities-by-peaks prefetch)
    CMD+=("${PREFETCH_ARGS[@]}")
else
    CMD=(
        sbatch
        --partition="$PARTITION"
        --qos="$QOS"
        --time="$TIME"
        --job-name="$PREFETCH_JOB_NAME"
        --output="$LOGS_DIR/prefetch_${PRESET}_%j.out"
        --error="$LOGS_DIR/prefetch_${PRESET}_%j.err"
    )
    if [ "$WAIT" = true ]; then
        CMD+=(--wait)
    fi
    CMD+=(slurm/lrc/prefetch_job.sh)
    CMD+=("${PREFETCH_ARGS[@]}")
fi

if [ "$DRY_RUN" = true ]; then
    printf '[DRY RUN] '
    printf '%q ' "${CMD[@]}"
    printf '\n'
else
    "${CMD[@]}"
fi
