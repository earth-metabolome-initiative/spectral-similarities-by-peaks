#!/bin/bash
# Submit a sharded finalize: an 18-task array followed by a dependent merge.

set -euo pipefail

REPO_DIR="${SPECTRAL_REPO_DIR:-$HOME/spectral-similarities-by-peaks}"
SCRATCH_ROOT="${SPECTRAL_SCRATCH_ROOT:-/global/scratch/users/$USER/spectral-similarities-by-peaks}"
LOGS_DIR="$SCRATCH_ROOT/logs"
TOTAL_CONFIGS=18

usage() {
    cat <<'USAGE'
Usage:
  bash slurm/lrc/finalize.sh <harmonized|gems> [OPTIONS]

Submits an 18-task array (one shard per similarity config) and a single
dependent merge job that concatenates the per-config outputs into the
canonical top-level artifacts.

Options:
  --partition=PART              SLURM partition (default: lr6)
  --qos=QOS                     SLURM QoS (default: lr_normal)
  --shard-time=HH:MM:SS         Wall time per shard (default: 06:00:00)
  --merge-time=HH:MM:SS         Wall time for the merge job (default: 06:00:00)
  --concurrency=N               Max concurrent shard tasks (default: 18)
  --data-dir=PATH               Dataset cache directory (default: data)
  --output-dir=PATH             Output directory for checkpoints/results
  --neighbors=N                 Top non-self neighbors (default: 64)
  --mz-tolerance=DA             Product m/z tolerance (default: 0.05)
  --row-sample-size=N           GeMS query-row sample size
  --reference-sample-size=N     GeMS reference-column sample size
  --keep-shard-dir              Retain _finalize_shards/ after merge for debugging
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
SHARD_TIME="06:00:00"
MERGE_TIME="06:00:00"
CONCURRENCY="$TOTAL_CONFIGS"
DATA_DIR="data"
OUTPUT_DIR=""
NEIGHBORS=64
MZ_TOLERANCE=0.05
ROW_SAMPLE_SIZE=""
REFERENCE_SAMPLE_SIZE=""
KEEP_SHARD_DIR=false
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
        --shard-time=*)             SHARD_TIME="${arg#*=}" ;;
        --merge-time=*)             MERGE_TIME="${arg#*=}" ;;
        --concurrency=*)            CONCURRENCY="${arg#*=}" ;;
        --data-dir=*)               DATA_DIR="${arg#*=}" ;;
        --output-dir=*)             OUTPUT_DIR="${arg#*=}" ;;
        --neighbors=*)              NEIGHBORS="${arg#*=}" ;;
        --mz-tolerance=*)           MZ_TOLERANCE="${arg#*=}" ;;
        --row-sample-size=*)        ROW_SAMPLE_SIZE="${arg#*=}" ;;
        --reference-sample-size=*)  REFERENCE_SAMPLE_SIZE="${arg#*=}" ;;
        --keep-shard-dir)           KEEP_SHARD_DIR=true ;;
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

MERGE_ARGS=("${SCAN_ARGS[@]}")
if [ "$KEEP_SHARD_DIR" = true ]; then
    MERGE_ARGS+=(--keep-shard-dir)
fi

ARRAY_RANGE="0-$((TOTAL_CONFIGS - 1))%$CONCURRENCY"

SHARD_JOB_NAME="spectral-$PRESET-finalize-shard"
MERGE_JOB_NAME="spectral-$PRESET-finalize-merge"

SHARD_CMD=(
    sbatch
    --parsable
    --partition="$PARTITION"
    --qos="$QOS"
    --time="$SHARD_TIME"
    --job-name="$SHARD_JOB_NAME"
    --array="$ARRAY_RANGE"
    --output="$LOGS_DIR/finalize_shard_${PRESET}_%A_%a.out"
    --error="$LOGS_DIR/finalize_shard_${PRESET}_%A_%a.err"
    slurm/lrc/finalize_shard_job.sh
)
SHARD_CMD+=("${SCAN_ARGS[@]}")

echo "=== Sharded finalize submission ==="
echo "Preset:       $PRESET"
echo "Output dir:   $OUTPUT_DIR"
echo "Configs:      $TOTAL_CONFIGS  (array $ARRAY_RANGE)"
echo "Partition:    $PARTITION"
echo "QoS:          $QOS"
echo "Shard time:   $SHARD_TIME"
echo "Merge time:   $MERGE_TIME"
echo "Logs:         $LOGS_DIR"
echo ""

if [ "$DRY_RUN" = true ]; then
    printf '[DRY RUN] '
    printf '%q ' "${SHARD_CMD[@]}"
    printf '\n'
    printf '[DRY RUN] sbatch --dependency=afterok:<arrayid> '
    printf '%q ' --partition="$PARTITION" --qos="$QOS" --time="$MERGE_TIME" \
        --job-name="$MERGE_JOB_NAME" \
        slurm/lrc/finalize_merge_job.sh "${MERGE_ARGS[@]}"
    printf '\n'
    exit 0
fi

ARRAY_ID=$("${SHARD_CMD[@]}")
echo "Submitted shard array job: $ARRAY_ID"

MERGE_CMD=(
    sbatch
    --parsable
    --dependency="afterok:$ARRAY_ID"
    --partition="$PARTITION"
    --qos="$QOS"
    --time="$MERGE_TIME"
    --job-name="$MERGE_JOB_NAME"
    --output="$LOGS_DIR/finalize_merge_${PRESET}_%j.out"
    --error="$LOGS_DIR/finalize_merge_${PRESET}_%j.err"
    slurm/lrc/finalize_merge_job.sh
)
MERGE_CMD+=("${MERGE_ARGS[@]}")

MERGE_ID=$("${MERGE_CMD[@]}")
echo "Submitted merge job: $MERGE_ID (waits for array $ARRAY_ID)"
