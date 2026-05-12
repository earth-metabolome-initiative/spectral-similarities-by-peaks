#!/bin/bash
# Download the selected dataset once before Lawrencium shard arrays start.
#SBATCH --job-name=spectral-prefetch
#SBATCH --account=pc_reese2026pi
#SBATCH --partition=lr6
#SBATCH --qos=lr_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/global/scratch/users/%u/spectral-similarities-by-peaks/logs/prefetch_%j.out
#SBATCH --error=/global/scratch/users/%u/spectral-similarities-by-peaks/logs/prefetch_%j.err

set -euo pipefail

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

export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"

echo "Host:    $(hostname)"
echo "Start:   $(date)"
echo "Command: target/release/spectral-similarities-by-peaks prefetch $*"

target/release/spectral-similarities-by-peaks prefetch "$@"

echo "Done:    $(date)"
