#!/bin/bash
#SBATCH --job-name=spectral-finalize
#SBATCH --account=ac_scscollab
#SBATCH --partition=lr6
#SBATCH --qos=lr_normal
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/global/scratch/users/%u/spectral-similarities-by-peaks/logs/finalize_%j.out
#SBATCH --error=/global/scratch/users/%u/spectral-similarities-by-peaks/logs/finalize_%j.err

set -euo pipefail

REPO_DIR="${SPECTRAL_REPO_DIR:-$HOME/spectral-similarities-by-peaks}"

clean_rust_compiler_environment() {
    unset CC CXX AR CFLAGS CXXFLAGS LDFLAGS
    unset CC_x86_64_unknown_linux_gnu CXX_x86_64_unknown_linux_gnu AR_x86_64_unknown_linux_gnu
    unset CFLAGS_x86_64_unknown_linux_gnu CXXFLAGS_x86_64_unknown_linux_gnu
}

clean_rust_compiler_environment
cd "$REPO_DIR"

if [ ! -x target/release/spectral-similarities-by-peaks ]; then
    echo "ERROR: release binary is missing. Run bash slurm/lrc/setup_env.sh first."
    exit 1
fi

export RAYON_NUM_THREADS="${SLURM_CPUS_ON_NODE:-$(nproc)}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"

echo "Host:       $(hostname)"
echo "Start:      $(date)"
echo "Rayon CPUs: $RAYON_NUM_THREADS"
echo "Command:    target/release/spectral-similarities-by-peaks finalize-scan $*"

target/release/spectral-similarities-by-peaks finalize-scan "$@"

echo "Done:       $(date)"
