#!/bin/bash
# Submit a single-node re-encoding pass that rewrites every `.parquet`
# under a results directory on Lawrencium using this crate's default
# zstd compression (via the `re-encode-parquets` CLI subcommand). The
# files stay valid parquet (random columnar access preserved), so the
# downstream readers don't need to change.

set -euo pipefail

REPO_DIR="${SPECTRAL_REPO_DIR:-$HOME/spectral-similarities-by-peaks}"
SCRATCH_ROOT="${SPECTRAL_SCRATCH_ROOT:-/global/scratch/users/$USER/spectral-similarities-by-peaks}"
LOGS_DIR="$SCRATCH_ROOT/logs"

usage() {
    cat <<'USAGE'
Usage:
  bash slurm/lrc/compress_parquets.sh <harmonized|gems|PATH> [OPTIONS]

Options:
  --partition=PART      SLURM partition (default: lr6)
  --qos=QOS             SLURM QoS       (default: lr_normal)
  --time=HH:MM:SS       Wall time       (default: 12:00:00)
  --level=N             zstd compression level (default: 22)
  --dry-run             Print the sbatch invocation without submitting
USAGE
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

TARGET="$1"
shift

PARTITION="lr6"
QOS="lr_normal"
TIME="12:00:00"
LEVEL="22"
DRY_RUN=false

case "$TARGET" in
    harmonized) ROOT="$SCRATCH_ROOT/results/harmonized-full" ;;
    gems)       ROOT="$SCRATCH_ROOT/results/gems-sampled"   ;;
    -h|--help)  usage; exit 0 ;;
    /*)         ROOT="$TARGET" ;;
    *)          ROOT="$REPO_DIR/$TARGET" ;;
esac

for arg in "$@"; do
    case "$arg" in
        --partition=*) PARTITION="${arg#*=}" ;;
        --qos=*)       QOS="${arg#*=}"       ;;
        --time=*)      TIME="${arg#*=}"      ;;
        --level=*)     LEVEL="${arg#*=}"     ;;
        --dry-run)     DRY_RUN=true          ;;
        -h|--help)     usage; exit 0         ;;
        *)             echo "Unknown option: $arg"; usage; exit 1 ;;
    esac
done

mkdir -p "$LOGS_DIR"
cd "$REPO_DIR"

CMD=(
    sbatch
    --partition="$PARTITION"
    --qos="$QOS"
    --time="$TIME"
    --job-name="spectral-compress-$(basename "$ROOT")"
    --output="$LOGS_DIR/compress_parquets_$(basename "$ROOT")_%j.out"
    --error="$LOGS_DIR/compress_parquets_$(basename "$ROOT")_%j.err"
    slurm/lrc/compress_parquets_job.sh
    "$ROOT"
)
# `--level` is preserved for compatibility but ignored: the actual
# compression level is now defined once inside the Rust binary's
# `parquet_writer_props()`. Suppress the unused-variable warning.
: "$LEVEL"

if [ "$DRY_RUN" = true ]; then
    printf '[DRY RUN] '
    printf '%q ' "${CMD[@]}"
    printf '\n'
else
    "${CMD[@]}"
fi
