#!/bin/bash
# Show Lawrencium shard progress for spectral-similarities-by-peaks.

set -euo pipefail

REPO_DIR="${SPECTRAL_REPO_DIR:-$HOME/spectral-similarities-by-peaks}"
SCRATCH_ROOT="${SPECTRAL_SCRATCH_ROOT:-/global/scratch/users/$USER/spectral-similarities-by-peaks}"
LOGS_DIR="$SCRATCH_ROOT/logs"
TOTAL_SHARDS=2304

usage() {
    cat <<'USAGE'
Usage:
  bash slurm/lrc/status.sh <harmonized|gems> [INTERVAL]
USAGE
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

PRESET="$1"
INTERVAL="${2:-}"

case "$PRESET" in
    harmonized) OUTPUT_DIR="results/harmonized-full" ;;
    gems)       OUTPUT_DIR="results/gems-sampled" ;;
    -h|--help)  usage; exit 0 ;;
    *)          echo "Unknown preset: $PRESET"; usage; exit 1 ;;
esac

SHARD_JOB_NAME="spectral-$PRESET"
PREFETCH_JOB_NAME="spectral-$PRESET-prefetch"

CONFIGS=(
    cosine_mz0.000_int1.000
    modified_cosine_mz0.000_int1.000
    cosine_mz1.000_int1.000
    modified_cosine_mz1.000_int1.000
    cosine_mz0.000_int0.500
    modified_cosine_mz0.000_int0.500
    cosine_mz1.000_int0.500
    modified_cosine_mz1.000_int0.500
    cosine_mz0.000_int0.250
    modified_cosine_mz0.000_int0.250
    cosine_mz1.000_int0.250
    modified_cosine_mz1.000_int0.250
    cosine_mz3.000_int0.600
    modified_cosine_mz3.000_int0.600
    entropy_mz0.000_int1.000_weightedtrue
    modified_entropy_mz0.000_int1.000_weightedtrue
    entropy_mz0.000_int1.000_weightedfalse
    modified_entropy_mz0.000_int1.000_weightedfalse
)

cd "$REPO_DIR"

show_status() {
    if [ -n "$INTERVAL" ]; then
        clear
    fi
    echo "=== spectral-similarities-by-peaks: $PRESET === ($(date '+%H:%M:%S'))"
    echo "Output dir: $OUTPUT_DIR"
    echo "Logs:       $LOGS_DIR"

    local completed=0
    if [ -d "$OUTPUT_DIR/distributions" ]; then
        completed=$(find "$OUTPUT_DIR/distributions" -name 'top_*.bincode.zst' -type f | wc -l)
    fi
    echo "Completed distribution shards: $completed / $TOTAL_SHARDS"
    if [ "$TOTAL_SHARDS" -gt 0 ]; then
        echo "Progress: $((completed * 100 / TOTAL_SHARDS))%"
    fi

    echo ""
    echo "First missing shards:"
    local shown=0
    for config in "${CONFIGS[@]}"; do
        for peak_count in $(seq 1 128); do
            local path
            path="$OUTPUT_DIR/distributions/$config/top_$(printf '%03d' "$peak_count").bincode.zst"
            if [ ! -f "$path" ]; then
                echo "  $config top $peak_count"
                shown=$((shown + 1))
                if [ "$shown" -ge 10 ]; then
                    break 2
                fi
            fi
        done
    done
    if [ "$shown" -eq 0 ]; then
        echo "  none"
    fi

    echo ""
    echo "=== SLURM queue for $PRESET ==="
    printf "%-12s %-24s %-8s %-10s %-6s %-4s %-12s %s\n" \
        JOBID NAME STATE TIME NODES CPUS PARTITION REASON
    local queue_rows
    queue_rows=$(squeue -h -u "$USER" -o "%.12i %.24j %.8T %.10M %.6D %.4C %.12P %R" 2>/dev/null \
        | awk -v shard="$SHARD_JOB_NAME" -v prefetch="$PREFETCH_JOB_NAME" \
            '$2 == shard || $2 == prefetch' || true)
    if [ -z "$queue_rows" ]; then
        echo "none"
    else
        echo "$queue_rows"
    fi
    local legacy_rows
    legacy_rows=$(squeue -h -u "$USER" -n spectral-shard \
        -o "%.12i %.24j %.8T %.10M %.6D %.4C %.12P %R" 2>/dev/null || true)
    if [ -n "$legacy_rows" ]; then
        echo ""
        echo "Legacy generic spectral-shard jobs are also running; they cannot be split by preset:"
        echo "$legacy_rows"
    fi

    echo ""
    echo "=== Recent non-empty errors ==="
    if [ -d "$LOGS_DIR" ]; then
        local err_files
        err_files=$(find "$LOGS_DIR" \
            \( -name "worker_${PRESET}_*.err" -o -name "prefetch_${PRESET}_*.err" \) \
            -size +0c -printf '%T@ %p\n' 2>/dev/null \
            | sort -rn | head -5 | awk '{print $2}')
        if [ -z "$err_files" ]; then
            echo "none"
        else
            for file in $err_files; do
                echo "--- $(basename "$file") ---"
                tail -8 "$file"
            done
        fi
    else
        echo "logs directory not found"
    fi
}

if [ -z "$INTERVAL" ]; then
    show_status
else
    while true; do
        show_status
        sleep "$INTERVAL"
    done
fi
