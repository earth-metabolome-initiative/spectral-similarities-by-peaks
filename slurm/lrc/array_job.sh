#!/bin/bash
#SBATCH --job-name=spectral-shard
#SBATCH --account=pc_reese2026pi
#SBATCH --partition=lr6
#SBATCH --qos=lr_normal
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/global/scratch/users/%u/spectral-similarities-by-peaks/logs/worker_%A_%a.out
#SBATCH --error=/global/scratch/users/%u/spectral-similarities-by-peaks/logs/worker_%A_%a.err

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch slurm/lrc/array_job.sh <SHARD_OFFSET> [SCAN_ARGS...]"
    exit 1
fi

SHARD_OFFSET="$1"
shift
SHARD_INDEX=$((SLURM_ARRAY_TASK_ID + SHARD_OFFSET))
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
    echo "ERROR: release binary is missing. Run bash slurm/lrc/setup_env.sh first."
    exit 1
fi

export RAYON_NUM_THREADS="${SLURM_CPUS_ON_NODE:-$(nproc)}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
export SPECTRAL_SIMILARITIES_FONT="${SPECTRAL_SIMILARITIES_FONT:-$HOME/fonts/DejaVuSans.ttf}"

echo "Host:        $(hostname)"
echo "Start:       $(date)"
echo "Shard index: $SHARD_INDEX"
echo "Rayon CPUs:  $RAYON_NUM_THREADS"
echo "Command:     target/release/spectral-similarities-by-peaks scan-shard $* --shard-index $SHARD_INDEX"

target/release/spectral-similarities-by-peaks scan-shard "$@" --shard-index "$SHARD_INDEX"

echo "Done:        $(date)"
