#!/bin/bash
# Submit spectral-similarities-by-peaks shard arrays on Lawrencium.

set -euo pipefail

REPO_DIR="${SPECTRAL_REPO_DIR:-$HOME/spectral-similarities-by-peaks}"
SCRATCH_ROOT="${SPECTRAL_SCRATCH_ROOT:-/global/scratch/users/$USER/spectral-similarities-by-peaks}"
LOGS_DIR="$SCRATCH_ROOT/logs"
TOTAL_SHARDS=2304
MAX_ARRAY_SIZE=1000

usage() {
    cat <<'USAGE'
Usage:
  bash slurm/lrc/submit.sh <harmonized|gems> [OPTIONS]

Options:
  --partition=PART              SLURM partition (default: lr6)
  --qos=QOS                     SLURM QoS (default: lr_normal)
  --time=HH:MM:SS               Wall time per shard (default: 24:00:00)
  --concurrency=N               Max concurrent array tasks (default: 64)
  --offset=N                    First zero-based shard index (default: 0)
  --n-shards=N                  Number of shard indexes to submit
  --data-dir=PATH               Dataset cache directory (default: data)
  --output-dir=PATH             Output directory for checkpoints/results
  --neighbors=N                 Top non-self neighbors (default: 64)
  --mz-tolerance=DA             Product m/z tolerance (default: 0.05)
  --row-sample-size=N           GeMS query-row sample size
  --reference-sample-size=N     GeMS reference-column sample size
  --gems-parts=N[,N...]         GeMS part numbers to load
  --prefetch-time=HH:MM:SS      Wall time for dataset prefetch (default: 06:00:00)
  --no-prefetch                 Submit shard arrays without a prefetch dependency
  --debug                       One lr_debug shard
  --dry-run                     Print sbatch commands without submitting
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
TIME="24:00:00"
CONCURRENCY=64
OFFSET=0
N_SHARDS=""
DATA_DIR="data"
OUTPUT_DIR=""
NEIGHBORS=64
MZ_TOLERANCE=0.05
ROW_SAMPLE_SIZE=""
REFERENCE_SAMPLE_SIZE=""
GEMS_PARTS=""
PREFETCH=true
PREFETCH_TIME="06:00:00"
DEBUG=false
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
        --concurrency=*)            CONCURRENCY="${arg#*=}" ;;
        --offset=*)                 OFFSET="${arg#*=}" ;;
        --n-shards=*)               N_SHARDS="${arg#*=}" ;;
        --data-dir=*)               DATA_DIR="${arg#*=}" ;;
        --output-dir=*)             OUTPUT_DIR="${arg#*=}" ;;
        --neighbors=*)              NEIGHBORS="${arg#*=}" ;;
        --mz-tolerance=*)           MZ_TOLERANCE="${arg#*=}" ;;
        --row-sample-size=*)        ROW_SAMPLE_SIZE="${arg#*=}" ;;
        --reference-sample-size=*)  REFERENCE_SAMPLE_SIZE="${arg#*=}" ;;
        --gems-parts=*)             GEMS_PARTS="${arg#*=}" ;;
        --prefetch-time=*)          PREFETCH_TIME="${arg#*=}" ;;
        --no-prefetch)              PREFETCH=false ;;
        --debug)                    DEBUG=true ;;
        --dry-run)                  DRY_RUN=true ;;
        -h|--help)                  usage; exit 0 ;;
        *)                          echo "Unknown option: $arg"; usage; exit 1 ;;
    esac
done

if [ "$DEBUG" = true ]; then
    PARTITION="lr4"
    QOS="lr_debug"
    TIME="03:00:00"
    CONCURRENCY=1
    N_SHARDS=1
fi

if [ -z "$N_SHARDS" ]; then
    N_SHARDS=$((TOTAL_SHARDS - OFFSET))
fi
if [ "$N_SHARDS" -le 0 ]; then
    echo "No shards to submit."
    exit 0
fi
if [ $((OFFSET + N_SHARDS)) -gt "$TOTAL_SHARDS" ]; then
    echo "ERROR: offset + n-shards exceeds $TOTAL_SHARDS total shards."
    exit 1
fi

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
if [ -n "$GEMS_PARTS" ]; then
    SCAN_ARGS+=(--gems-parts "$GEMS_PARTS")
fi

PREFETCH_ARGS=(
    --dataset "$PRESET"
    --data-dir "$DATA_DIR"
)
if [ -n "$GEMS_PARTS" ]; then
    PREFETCH_ARGS+=(--gems-parts "$GEMS_PARTS")
fi

SHARD_JOB_NAME="spectral-$PRESET"
PREFETCH_JOB_NAME="spectral-$PRESET-prefetch"

echo "=== Lawrencium shard submission ==="
echo "Preset:       $PRESET"
echo "Output dir:   $OUTPUT_DIR"
echo "Shard range:  $OFFSET..$((OFFSET + N_SHARDS - 1)) / $TOTAL_SHARDS"
echo "Partition:    $PARTITION"
echo "QoS:          $QOS"
echo "Time:         $TIME"
echo "Concurrency:  $CONCURRENCY"
echo "Prefetch:     $PREFETCH"
echo "Logs:         $LOGS_DIR"
echo ""

DEPENDENCY=""
if [ "$PREFETCH" = true ]; then
    PREFETCH_CMD=(
        sbatch
        --partition="$PARTITION"
        --qos="$QOS"
        --time="$PREFETCH_TIME"
        --job-name="$PREFETCH_JOB_NAME"
        --output="$LOGS_DIR/prefetch_${PRESET}_%j.out"
        --error="$LOGS_DIR/prefetch_${PRESET}_%j.err"
        slurm/lrc/prefetch_job.sh
    )
    PREFETCH_CMD+=("${PREFETCH_ARGS[@]}")

    if [ "$DRY_RUN" = true ]; then
        printf '[DRY RUN] '
        printf '%q ' "${PREFETCH_CMD[@]}"
        printf '\n'
        DEPENDENCY="afterok:<prefetch-job-id>"
    else
        echo "Submitting dataset prefetch"
        PREFETCH_OUTPUT="$("${PREFETCH_CMD[@]}")"
        echo "$PREFETCH_OUTPUT"
        if [[ "$PREFETCH_OUTPUT" =~ Submitted[[:space:]]batch[[:space:]]job[[:space:]]([0-9]+) ]]; then
            DEPENDENCY="afterok:${BASH_REMATCH[1]}"
        else
            echo "ERROR: could not parse prefetch job id from sbatch output."
            exit 1
        fi
    fi
fi

SUBMITTED=0
while [ "$SUBMITTED" -lt "$N_SHARDS" ]; do
    BATCH_SIZE=$((N_SHARDS - SUBMITTED))
    if [ "$BATCH_SIZE" -gt "$MAX_ARRAY_SIZE" ]; then
        BATCH_SIZE="$MAX_ARRAY_SIZE"
    fi
    BATCH_MAX=$((BATCH_SIZE - 1))
    BATCH_OFFSET=$((OFFSET + SUBMITTED))

    SBATCH_CMD=(
        sbatch
        --partition="$PARTITION"
        --qos="$QOS"
        --time="$TIME"
        --job-name="$SHARD_JOB_NAME"
        --array="0-${BATCH_MAX}%${CONCURRENCY}"
        --output="$LOGS_DIR/worker_${PRESET}_%A_%a.out"
        --error="$LOGS_DIR/worker_${PRESET}_%A_%a.err"
    )
    if [ -n "$DEPENDENCY" ]; then
        SBATCH_CMD+=(--dependency="$DEPENDENCY")
    fi
    SBATCH_CMD+=(
        slurm/lrc/array_job.sh
        "$BATCH_OFFSET"
    )
    SBATCH_CMD+=("${SCAN_ARGS[@]}")

    if [ "$DRY_RUN" = true ]; then
        printf '[DRY RUN] '
        printf '%q ' "${SBATCH_CMD[@]}"
        printf '\n'
    else
        echo "Submitting batch offset=$BATCH_OFFSET size=$BATCH_SIZE"
        "${SBATCH_CMD[@]}"
    fi

    SUBMITTED=$((SUBMITTED + BATCH_SIZE))
done
