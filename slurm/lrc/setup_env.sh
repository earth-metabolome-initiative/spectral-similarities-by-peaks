#!/bin/bash
# One-time Lawrencium setup for spectral-similarities-by-peaks.

set -euo pipefail

REPO_URL="${SPECTRAL_REPO_URL:-https://github.com/earth-metabolome-initiative/spectral-similarities-by-peaks.git}"
REPO_DIR="${SPECTRAL_REPO_DIR:-$HOME/spectral-similarities-by-peaks}"
SCRATCH_ROOT="${SPECTRAL_SCRATCH_ROOT:-/global/scratch/users/$USER/spectral-similarities-by-peaks}"
DATA_DIR="$SCRATCH_ROOT/data"
RESULTS_DIR="$SCRATCH_ROOT/results"
LOGS_DIR="$SCRATCH_ROOT/logs"
MIN_RUST_VERSION="1.86.0"

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

try_load_rust_module() {
    if command -v module > /dev/null 2>&1; then
        module load rust > /dev/null 2>&1 || true
    fi
}

install_rustup() {
    local installer
    installer="$(mktemp)"
    echo "Installing Rust with rustup under $HOME/.cargo"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o "$installer"
    sh "$installer" -y --profile minimal --default-toolchain stable
    rm -f "$installer"
    load_user_cargo_environment
}

cargo_version() {
    local output version
    output="$(cargo --version 2> /dev/null)" || return 1
    version="${output#cargo }"
    printf '%s\n' "${version%% *}"
}

version_ge() {
    local candidate="$1"
    local required="$2"
    local candidate_major candidate_minor candidate_patch
    local required_major required_minor required_patch

    IFS=. read -r candidate_major candidate_minor candidate_patch <<< "$candidate"
    IFS=. read -r required_major required_minor required_patch <<< "$required"
    candidate_patch="${candidate_patch%%[^0-9]*}"
    required_patch="${required_patch%%[^0-9]*}"

    candidate_major="${candidate_major:-0}"
    candidate_minor="${candidate_minor:-0}"
    candidate_patch="${candidate_patch:-0}"
    required_major="${required_major:-0}"
    required_minor="${required_minor:-0}"
    required_patch="${required_patch:-0}"

    if [ "$candidate_major" -ne "$required_major" ]; then
        [ "$candidate_major" -gt "$required_major" ]
        return
    fi
    if [ "$candidate_minor" -ne "$required_minor" ]; then
        [ "$candidate_minor" -gt "$required_minor" ]
        return
    fi
    [ "$candidate_patch" -ge "$required_patch" ]
}

cargo_is_recent_enough() {
    local version
    command -v cargo > /dev/null 2>&1 || return 1
    version="$(cargo_version)" || return 1
    version_ge "$version" "$MIN_RUST_VERSION"
}

install_or_update_rustup_stable() {
    if ! command -v rustup > /dev/null 2>&1; then
        install_rustup
    fi

    rustup toolchain install stable --profile minimal
    rustup default stable
    load_user_cargo_environment
}

ensure_cargo() {
    load_user_cargo_environment
    if cargo_is_recent_enough; then
        return
    fi

    try_load_rust_module
    if cargo_is_recent_enough; then
        return
    fi

    if command -v cargo > /dev/null 2>&1; then
        echo "Found $(cargo --version), but this project requires Rust/Cargo $MIN_RUST_VERSION+."
    fi

    install_or_update_rustup_stable
    if ! cargo_is_recent_enough; then
        echo "ERROR: cargo is still unavailable or older than $MIN_RUST_VERSION after rustup setup."
        exit 1
    fi
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
ensure_cargo

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

# Lawrencium compute nodes lack system font packages, so the heatmap renderer
# falls back to SPECTRAL_SIMILARITIES_FONT. Fetch DejaVu Sans into $HOME/fonts
# once and re-fetch if the cached copy is invalid. The SLURM job scripts
# default the env var to this path.
FONT_DIR="$HOME/fonts"
FONT_FILE="$FONT_DIR/DejaVuSans.ttf"
DEJAVU_ZIP_URL="https://downloads.sourceforge.net/project/dejavu/dejavu/2.37/dejavu-fonts-ttf-2.37.zip"
DEJAVU_ZIP_MEMBER="dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf"
FONT_MIN_BYTES=102400

is_valid_ttf() {
    local path="$1"
    [ -s "$path" ] || return 1
    local size
    size=$(stat -c '%s' "$path" 2>/dev/null || echo 0)
    [ "$size" -ge "$FONT_MIN_BYTES" ] || return 1
    local magic
    magic=$(head -c 4 "$path" | od -An -tx1 -N4 | tr -d ' \n')
    [ "$magic" = "00010000" ]
}

mkdir -p "$FONT_DIR"
if is_valid_ttf "$FONT_FILE"; then
    echo "DejaVu Sans already present and valid at $FONT_FILE"
else
    if [ -e "$FONT_FILE" ]; then
        echo "Replacing invalid font at $FONT_FILE"
        rm -f "$FONT_FILE"
    fi
    echo "Fetching DejaVu Sans into $FONT_FILE"
    tmp_zip="$(mktemp)"
    curl --proto '=https' --tlsv1.2 -fsSL -o "$tmp_zip" "$DEJAVU_ZIP_URL"
    unzip -p "$tmp_zip" "$DEJAVU_ZIP_MEMBER" > "$FONT_FILE"
    rm -f "$tmp_zip"
    if ! is_valid_ttf "$FONT_FILE"; then
        echo "ERROR: $FONT_FILE failed TTF validation after download" >&2
        rm -f "$FONT_FILE"
        exit 1
    fi
fi

cd "$REPO_DIR"
# Build a binary portable across Lawrencium partitions. The login node has
# newer CPU instructions than the lr4 debug nodes, so target-cpu=native breaks
# cross-partition runs with SIGILL. x86-64-v3 covers Haswell+ (AVX2) which is
# the floor across lr4, lr5, and lr6. Override with RUSTFLAGS if you really
# want a host-specific build.
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=x86-64-v3}"
rustc --version
cargo --version
cargo build --release --locked

target/release/spectral-similarities-by-peaks --help > /dev/null

echo "Setup complete."
echo "Repo:    $REPO_DIR"
echo "Data:    $DATA_DIR"
echo "Results: $RESULTS_DIR"
echo "Logs:    $LOGS_DIR"
echo "Submit:  bash slurm/lrc/submit.sh harmonized --dry-run"
