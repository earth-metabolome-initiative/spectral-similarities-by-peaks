#!/bin/bash
#SBATCH --job-name=spectral-compress-parquets
#SBATCH --account=pc_reese2026pi
#SBATCH --partition=lr6
#SBATCH --qos=lr_normal
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/global/scratch/users/%u/spectral-similarities-by-peaks/logs/compress_parquets_%j.out
#SBATCH --error=/global/scratch/users/%u/spectral-similarities-by-peaks/logs/compress_parquets_%j.err
#
# Re-encode every `.parquet` under the given results directory in place
# with this crate's default zstd compression. Output stays as valid
# parquet (random columnar access preserved). A 10-shard sample of
# pathway_scores at zstd-22 averaged 0.142 (85.8 % reduction) over the
# legacy Snappy-encoded files, so the cluster footprint should shrink
# accordingly and any subsequent transfer is much faster.
#
# Use:
#     sbatch slurm/lrc/compress_parquets_job.sh /global/scratch/users/$USER/spectral-similarities-by-peaks/results/harmonized-full

set -euo pipefail

ROOT="${1:-}"
if [ -z "$ROOT" ]; then
    echo "Usage: $0 <results-dir-on-cluster>" >&2
    exit 1
fi
if [ ! -d "$ROOT" ]; then
    echo "ERROR: $ROOT is not a directory" >&2
    exit 1
fi

REPO_DIR="${SPECTRAL_REPO_DIR:-$HOME/spectral-similarities-by-peaks}"

clean_rust_compiler_environment() {
    unset CC CXX AR CFLAGS CXXFLAGS LDFLAGS
    unset CC_x86_64_unknown_linux_gnu CXX_x86_64_unknown_linux_gnu AR_x86_64_unknown_linux_gnu
    unset CFLAGS_x86_64_unknown_linux_gnu CXXFLAGS_x86_64_unknown_linux_gnu
}

load_user_cargo_environment() {
    if [ -f "$HOME/.cargo/env" ]; then
        # shellcheck source=/dev/null
        . "$HOME/.cargo/env"
    fi
}

clean_rust_compiler_environment
load_user_cargo_environment
cd "$REPO_DIR"

if [ ! -x target/release/spectral-similarities-by-peaks ]; then
    echo "ERROR: release binary is missing. Run bash slurm/lrc/setup_env.sh first." >&2
    exit 1
fi

export RAYON_NUM_THREADS="${SLURM_CPUS_ON_NODE:-$(nproc)}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"

before=$(du -sb "$ROOT" | awk '{print $1}')
parquets=$(find "$ROOT" -name '*.parquet' | wc -l)

echo "Host:        $(hostname)"
echo "Start:       $(date)"
echo "Root:        $ROOT"
echo "Parquets:    $parquets"
echo "Rayon CPUs:  $RAYON_NUM_THREADS"

target/release/spectral-similarities-by-peaks re-encode-parquets --output-dir "$ROOT"

after=$(du -sb "$ROOT" | awk '{print $1}')
awk -v b="$before" -v a="$after" 'BEGIN {
    printf "Before:      %.2f GiB\n", b / 1073741824
    printf "After:       %.2f GiB\n", a / 1073741824
    if (b > 0) printf "Ratio:       %.3f (%.1f%% reduction)\n", a / b, (1 - a / b) * 100
}'
echo "Done:        $(date)"
