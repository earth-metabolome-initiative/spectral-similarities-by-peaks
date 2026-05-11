#!/bin/bash
# Cancel Lawrencium jobs for one preset and clean interrupted checkpoint writes.

set -euo pipefail

REPO_DIR="${SPECTRAL_REPO_DIR:-$HOME/spectral-similarities-by-peaks}"
WAIT_SECONDS=120
WAIT=true
CLEAN=true
DRY_RUN=false
INCLUDE_LEGACY=false
CUSTOM_OUTPUT_DIR=""

usage() {
    cat <<'USAGE'
Usage:
  bash slurm/lrc/cancel.sh <harmonized|gems|all> [OPTIONS]

Options:
  --include-legacy      Also cancel old generic spectral-shard jobs.
  --only-legacy         Cancel only old generic spectral-* jobs.
  --output-dir=PATH     Additional output directory to clean for temp files.
  --wait-seconds=N      Seconds to wait before cleanup (default: 120).
  --no-wait             Do not wait for matching jobs to disappear.
  --no-clean            Do not clean temporary checkpoint/artifact files.
  --dry-run             Print jobs/files without cancelling or deleting.
USAGE
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

TARGET="$1"
shift

case "$TARGET" in
    harmonized)
        JOB_NAMES=(spectral-harmonized spectral-harmonized-prefetch spectral-harmonized-finalize)
        OUTPUT_DIRS=(results/harmonized-full)
        ;;
    gems)
        JOB_NAMES=(spectral-gems spectral-gems-prefetch spectral-gems-finalize)
        OUTPUT_DIRS=(results/gems-sampled)
        ;;
    all)
        JOB_NAMES=(
            spectral-harmonized
            spectral-harmonized-prefetch
            spectral-harmonized-finalize
            spectral-gems
            spectral-gems-prefetch
            spectral-gems-finalize
        )
        OUTPUT_DIRS=(results/harmonized-full results/gems-sampled)
        INCLUDE_LEGACY=true
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown target: $TARGET"
        usage
        exit 1
        ;;
esac

for arg in "$@"; do
    case "$arg" in
        --include-legacy)    INCLUDE_LEGACY=true ;;
        --only-legacy)       JOB_NAMES=(); INCLUDE_LEGACY=true ;;
        --output-dir=*)      CUSTOM_OUTPUT_DIR="${arg#*=}" ;;
        --wait-seconds=*)    WAIT_SECONDS="${arg#*=}" ;;
        --no-wait)           WAIT=false ;;
        --no-clean)          CLEAN=false ;;
        --dry-run)           DRY_RUN=true ;;
        -h|--help)           usage; exit 0 ;;
        *)                   echo "Unknown option: $arg"; usage; exit 1 ;;
    esac
done

if [ "$INCLUDE_LEGACY" = true ]; then
    JOB_NAMES+=(spectral-shard spectral-prefetch spectral-finalize)
fi
if [ -n "$CUSTOM_OUTPUT_DIR" ]; then
    OUTPUT_DIRS+=("$CUSTOM_OUTPUT_DIR")
fi

cd "$REPO_DIR"

job_name_selected() {
    local candidate="$1"
    local job_name
    for job_name in "${JOB_NAMES[@]}"; do
        if [ "$candidate" = "$job_name" ]; then
            return 0
        fi
    done
    return 1
}

collect_job_ids() {
    local row job_id job_name
    local rows=()
    mapfile -t rows < <(squeue -h -u "$USER" -o "%i %j" 2>/dev/null || true)
    for row in "${rows[@]}"; do
        read -r job_id job_name <<< "$row"
        if job_name_selected "$job_name"; then
            printf '%s\n' "$job_id"
        fi
    done | sort -u
}

print_selected_jobs() {
    local ids=("$@")
    local id_list
    if [ "${#ids[@]}" -eq 0 ]; then
        echo "No matching SLURM jobs found."
        return
    fi
    id_list=$(IFS=,; printf '%s' "${ids[*]}")
    squeue -j "$id_list" -o "%.18i %.24j %.8T %.10M %.6D %.4C %.12P %R" 2>/dev/null || true
}

matching_jobs_remain() {
    local ids=()
    mapfile -t ids < <(collect_job_ids)
    [ "${#ids[@]}" -gt 0 ]
}

wait_for_jobs_to_leave_queue() {
    local deadline=$((SECONDS + WAIT_SECONDS))
    if [ "$WAIT" != true ]; then
        return 0
    fi
    while matching_jobs_remain; do
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "WARNING: matching jobs are still present after ${WAIT_SECONDS}s."
            return 1
        fi
        sleep 5
    done
    return 0
}

clean_output_dir() {
    local output_dir="$1"
    if [ ! -d "$output_dir" ]; then
        echo "No output directory: $output_dir"
        return
    fi

    echo "Temporary files in $output_dir:"
    if [ "$DRY_RUN" = true ]; then
        find "$output_dir" -type f \
            \( -name 'top_*.bincode.zst.tmp-*' \
            -o -name 'top_*.bincode.tmp-*' \
            -o -name '*.parquet.tmp-*' \) \
            -print
    else
        find "$output_dir" -type f \
            \( -name 'top_*.bincode.zst.tmp-*' \
            -o -name 'top_*.bincode.tmp-*' \
            -o -name '*.parquet.tmp-*' \) \
            -print -delete
    fi
}

mapfile -t JOB_IDS < <(collect_job_ids)

echo "=== Lawrencium cancellation ==="
echo "Target:         $TARGET"
echo "Job names:      ${JOB_NAMES[*]:-none}"
echo "Wait seconds:   $WAIT_SECONDS"
echo "Cleanup:        $CLEAN"
echo "Dry run:        $DRY_RUN"
echo ""

print_selected_jobs "${JOB_IDS[@]}"

if [ "${#JOB_IDS[@]}" -gt 0 ]; then
    if [ "$DRY_RUN" = true ]; then
        printf '[DRY RUN] scancel'
        printf ' %q' "${JOB_IDS[@]}"
        printf '\n'
    else
        scancel "${JOB_IDS[@]}"
    fi
fi

if [ "$DRY_RUN" != true ] && [ "${#JOB_IDS[@]}" -gt 0 ]; then
    if ! wait_for_jobs_to_leave_queue; then
        echo "Skipping cleanup while matching jobs remain."
        exit 1
    fi
fi

if [ "$CLEAN" = true ]; then
    for output_dir in "${OUTPUT_DIRS[@]}"; do
        clean_output_dir "$output_dir"
    done
fi
