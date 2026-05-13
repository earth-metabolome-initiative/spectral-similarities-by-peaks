#!/usr/bin/env bash
# Mirror Lawrencium results to a local backup directory using parallel rsync.
#
# Runs from a WORKSTATION (not the cluster). Pulls
#   <remote-base>/<preset>/
# to
#   <local-base>/<preset>/
# splitting the per-config subtrees under distributions/ and pathway_shards/
# across N parallel rsync streams to work around the per-SSH-channel
# bandwidth-delay cap on the long-haul link.
#
# Requires:
#   - rsync, ssh, parallel (GNU)
#   - An SSH ControlMaster to the remote host already alive (avoids re-auth
#     per worker). The script will print instructions and abort if it cannot
#     reach the remote in one round trip.

set -euo pipefail

# LRC's sshd injects the federal-system NOTICE banner on every session channel,
# which floods rsync progress output. -q silences server banners (real errors
# from rsync itself still surface on its own pipe).
export RSYNC_RSH="ssh -q"

REMOTE_USER=""        # filled in below from --remote-user or `ssh -G`
REMOTE_HOST="lrc-login.lbl.gov"
REMOTE_BASE=""
LOCAL_BASE="/mnt/bfd/spetral-similarity"
PARALLEL_JOBS=8
SKIP_PATHWAY_SHARDS=false
SKIP_PATHWAY_PARQUETS=false
DRY_RUN=false
PRESET=""

usage() {
    cat <<'USAGE'
Usage:
  bash slurm/lrc/pull_results.sh <PRESET> [OPTIONS]

PRESET:
  harmonized        mirror results/harmonized-full
  gems              mirror results/gems-sampled
  all               mirror both presets

Options:
  --remote-user=USER         Override SSH user (default: resolved from `ssh -G`,
                             which honors your ~/.ssh/config Host entry; falls
                             back to $USER if no User directive is set)
  --remote-host=HOST         Override SSH host (default: lrc-login.lbl.gov)
  --remote-base=PATH         Override remote results root
                             (default: /global/scratch/users/<user>/spectral-similarities-by-peaks/results)
  --local-base=PATH          Override local destination root
                             (default: /mnt/bfd/spetral-similarity)
  --parallel-jobs=N          Concurrent rsync workers per subtree (default: 8).
                             LRC's sshd MaxSessions is 10, so 8 workers leaves
                             headroom for `watch`/probe sessions. Push higher
                             only if you've opened additional ControlMasters or
                             confirmed the server allows more channels (sshd_config
                             MaxSessions). Symptoms of overshoot: "Session open
                             refused by peer" or "administratively prohibited".
  --skip-pathway-shards      Skip pathway_shards/ (hundreds of GB)
  --skip-pathway-parquets    Skip pathway_scores.parquet and pathway_predictions.parquet
  --dry-run                  Print actions without transferring
  -h, --help                 Show this help
USAGE
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

case "$1" in
    -h|--help) usage; exit 0 ;;
esac

PRESET="$1"
shift

for arg in "$@"; do
    case "$arg" in
        --remote-user=*)         REMOTE_USER="${arg#*=}" ;;
        --remote-host=*)         REMOTE_HOST="${arg#*=}" ;;
        --remote-base=*)         REMOTE_BASE="${arg#*=}" ;;
        --local-base=*)          LOCAL_BASE="${arg#*=}" ;;
        --parallel-jobs=*)       PARALLEL_JOBS="${arg#*=}" ;;
        --skip-pathway-shards)   SKIP_PATHWAY_SHARDS=true ;;
        --skip-pathway-parquets) SKIP_PATHWAY_PARQUETS=true ;;
        --dry-run)               DRY_RUN=true ;;
        -h|--help)               usage; exit 0 ;;
        *)                       echo "Unknown option: $arg" >&2; usage; exit 1 ;;
    esac
done

case "$PRESET" in
    harmonized) PRESETS=(harmonized-full) ;;
    gems)       PRESETS=(gems-sampled) ;;
    all)        PRESETS=(harmonized-full gems-sampled) ;;
    *)          echo "Unknown preset: $PRESET" >&2; usage; exit 1 ;;
esac

if [ -z "$REMOTE_USER" ]; then
    REMOTE_USER=$(ssh -G "$REMOTE_HOST" 2> /dev/null | awk '/^user / {print $2}')
fi
if [ -z "$REMOTE_USER" ]; then
    REMOTE_USER="$USER"
fi

if [ -z "$REMOTE_BASE" ]; then
    REMOTE_BASE="/global/scratch/users/$REMOTE_USER/spectral-similarities-by-peaks/results"
fi

for cmd in rsync ssh parallel; do
    if ! command -v "$cmd" > /dev/null 2>&1; then
        echo "ERROR: required command not in PATH: $cmd" >&2
        echo "Install with: sudo apt install -y rsync openssh-client parallel" >&2
        exit 1
    fi
done

REMOTE="$REMOTE_USER@$REMOTE_HOST"

echo "=== one-shot remote reachability check ==="
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR "$REMOTE" true; then
    cat <<EOF >&2

ERROR: cannot reach $REMOTE in one round trip without an interactive prompt.

This script assumes you already have an SSH ControlMaster to the remote host.
If not, open one in another terminal first:

    ssh -fN $REMOTE_HOST

Confirm it's alive with:

    ssh -O check $REMOTE_HOST

Then re-run this script.

EOF
    exit 1
fi

run() {
    if [ "$DRY_RUN" = true ]; then
        printf '[DRY RUN] '
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

rsync_subdir_listing() {
    local remote_dir="$1"
    ssh -q -o LogLevel=ERROR "$REMOTE" \
        "test -d $remote_dir && ls $remote_dir 2>/dev/null" \
        || true
}

mirror_per_config_tree() {
    local preset="$1"
    local subtree="$2"
    local remote_dir="$REMOTE_BASE/$preset/$subtree"
    local local_dir="$LOCAL_BASE/$preset/$subtree"

    local entries
    entries=$(rsync_subdir_listing "$remote_dir")
    if [ -z "$entries" ]; then
        echo "  (no $subtree/ contents on remote, skipping)"
        return
    fi

    mkdir -p "$local_dir"
    local exports
    exports=$(printf '%s\n' "$entries")
    # shellcheck disable=SC2016
    if [ "$DRY_RUN" = true ]; then
        printf '[DRY RUN] parallel -j %d rsync per subdir:\n' "$PARALLEL_JOBS"
        printf '%s\n' "$exports" | sed 's/^/  /'
    else
        # --line-buffer + --tag flush each rsync's progress line as it arrives,
        # prefixed by the config-dir name. Without them, parallel batches output
        # per-worker and the terminal looks frozen for the duration of each
        # transfer.
        printf '%s\n' "$exports" | parallel -j "$PARALLEL_JOBS" --line-buffer --tag \
            rsync -ahW --partial --info=progress2 \
            "$REMOTE:$remote_dir/{}/" "$local_dir/{}/"
    fi
}

mirror_top_level() {
    local preset="$1"
    local remote_dir="$REMOTE_BASE/$preset/"
    local local_dir="$LOCAL_BASE/$preset/"
    mkdir -p "$local_dir"

    local excludes=(
        --exclude '**/*.tmp-*'
        --exclude 'distributions/'
        --exclude 'pathway_shards/'
        --exclude 'distributions.stray-pre-fix/***'
    )
    if [ "$SKIP_PATHWAY_PARQUETS" = true ]; then
        excludes+=(--exclude 'pathway_scores.parquet')
        excludes+=(--exclude 'pathway_predictions.parquet')
    fi

    run rsync -ahW --partial --info=progress2 "${excludes[@]}" \
        "$REMOTE:$remote_dir" "$local_dir"
}

for preset in "${PRESETS[@]}"; do
    echo ""
    echo "================================================================"
    echo "Preset: $preset"
    echo "  Remote:  $REMOTE:$REMOTE_BASE/$preset/"
    echo "  Local:   $LOCAL_BASE/$preset/"
    echo "  Workers: $PARALLEL_JOBS"
    echo "================================================================"

    echo ""
    echo "[1/3] distributions/  (per-config, $PARALLEL_JOBS-way parallel)"
    mirror_per_config_tree "$preset" "distributions"

    if [ "$SKIP_PATHWAY_SHARDS" = true ]; then
        echo ""
        echo "[2/3] pathway_shards/  (skipped)"
    else
        echo ""
        echo "[2/3] pathway_shards/ (per-config, $PARALLEL_JOBS-way parallel)"
        mirror_per_config_tree "$preset" "pathway_shards"
    fi

    echo ""
    echo "[3/3] top-level files (Parquet, NPZ, heatmaps, pathway_prediction_*)"
    mirror_top_level "$preset"
done

echo ""
echo "=== Done ==="
for preset in "${PRESETS[@]}"; do
    local_dir="$LOCAL_BASE/$preset"
    distribution_count=$(find "$local_dir/distributions" -name 'top_*.bincode.zst' -type f 2>/dev/null | wc -l)
    heatmap_count=$(find "$local_dir/heatmaps" -name '*.png' -type f 2>/dev/null | wc -l)
    pathway_prediction_count=$(find "$local_dir/pathway_prediction_heatmaps" -name '*.png' -type f 2>/dev/null | wc -l)
    printf '%-20s  distributions=%d  heatmaps=%d  pathway_prediction_heatmaps=%d\n' \
        "$preset" "$distribution_count" "$heatmap_count" "$pathway_prediction_count"
done
