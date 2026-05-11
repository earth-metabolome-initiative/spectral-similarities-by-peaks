#!/bin/bash
# One-time Lawrencium setup for spectral-similarities-by-peaks.

set -euo pipefail

REPO_URL="${SPECTRAL_REPO_URL:-https://github.com/earth-metabolome-initiative/spectral-similarities-by-peaks.git}"
REPO_DIR="${SPECTRAL_REPO_DIR:-$HOME/spectral-similarities-by-peaks}"
SCRATCH_ROOT="${SPECTRAL_SCRATCH_ROOT:-/global/scratch/users/$USER/spectral-similarities-by-peaks}"
DATA_DIR="$SCRATCH_ROOT/data"
RESULTS_DIR="$SCRATCH_ROOT/results"
LOGS_DIR="$SCRATCH_ROOT/logs"

clean_rust_compiler_environment() {
    unset CC CXX AR CFLAGS CXXFLAGS LDFLAGS
    unset CC_x86_64_unknown_linux_gnu CXX_x86_64_unknown_linux_gnu AR_x86_64_unknown_linux_gnu
    unset CFLAGS_x86_64_unknown_linux_gnu CXXFLAGS_x86_64_unknown_linux_gnu
}

link_scratch_directory() {
    local link_path="$1"
    local target_path="$2"

    if [ -L "$link_path" ]; then
        echo "$link_path already points to $(readlink "$link_path")"
    elif [ -e "$link_path" ]; then
        echo "ERROR: $link_path exists and is not a symlink."
        echo "Move it manually before rerunning this setup."
        exit 1
    else
        ln -s "$target_path" "$link_path"
        echo "Created $link_path -> $target_path"
    fi
}

clean_rust_compiler_environment

if [ -d "$REPO_DIR/.git" ]; then
    echo "Updating $REPO_DIR"
    git -C "$REPO_DIR" pull --ff-only
elif [ -e "$REPO_DIR" ]; then
    echo "ERROR: $REPO_DIR exists but is not a git checkout."
    exit 1
else
    echo "Cloning $REPO_URL into $REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
fi

mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$LOGS_DIR"
link_scratch_directory "$REPO_DIR/data" "$DATA_DIR"
link_scratch_directory "$REPO_DIR/results" "$RESULTS_DIR"
link_scratch_directory "$REPO_DIR/logs" "$LOGS_DIR"

cd "$REPO_DIR"
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
cargo build --release --locked

target/release/spectral-similarities-by-peaks --help > /dev/null

echo "Setup complete."
echo "Repo:    $REPO_DIR"
echo "Data:    $DATA_DIR"
echo "Results: $RESULTS_DIR"
echo "Logs:    $LOGS_DIR"
echo "Submit:  bash slurm/lrc/submit.sh harmonized --dry-run"
