#!/usr/bin/env bash
# Deploy + benchmark powelleem on a Linux box where fork-based multiprocessing
# works natively (unlike macOS Accelerate which crashes after fork).
#
# Usage:
#   # From your Mac:
#   scp deploy/run_on_linux.sh <user@host>:~/
#   ssh <user@host> 'bash run_on_linux.sh'
#
# The script:
#   1. Installs miniconda if missing (skip if already installed)
#   2. Creates a fresh env `powelleem` with Python 3.11
#   3. Clones github.com/guillaume-osmo/powelleem
#   4. Installs the package + dev/powell extras
#   5. Downloads the NEEMP example datasets from the published release (TODO)
#   6. Runs the full benchmark and dumps results
#
# Tested on Ubuntu 22.04+ and Debian 12 with miniconda3.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/guillaume-osmo/powelleem.git}"
WORK_DIR="${WORK_DIR:-$HOME/powelleem-bench}"
N_WORKERS="${N_WORKERS:-$(nproc)}"
NEEMP_DIR="${NEEMP_DIR:-$HOME/neemp-data}"  # expect set01.{sdf,chg,typ} etc. here

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- 1. miniconda ----------
if ! command -v conda &>/dev/null; then
    log "Installing miniconda…"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
else
    eval "$(conda shell.bash hook)"
fi

# ---------- 2. env ----------
if ! conda env list | grep -q "^powelleem "; then
    log "Creating conda env 'powelleem'…"
    conda create -y -n powelleem python=3.11
fi
conda activate powelleem

# ---------- 3. clone repo ----------
if [[ ! -d "$WORK_DIR" ]]; then
    log "Cloning $REPO_URL…"
    git clone "$REPO_URL" "$WORK_DIR"
else
    log "Updating $WORK_DIR…"
    (cd "$WORK_DIR" && git pull --ff-only)
fi
cd "$WORK_DIR"

# ---------- 4. install ----------
log "Installing powelleem[dev,powell,rdkit]…"
pip install --upgrade pip
pip install -e ".[dev,powell,rdkit]"

# ---------- 5. dataset check ----------
if [[ ! -f "$NEEMP_DIR/set01.sdf" ]]; then
    fail "NEEMP datasets not found at $NEEMP_DIR. Please scp the de-uoa-matlab/neemp/examples/ directory to $NEEMP_DIR."
fi

# Point the bench scripts at the local NEEMP_DIR
export POWELLEEM_NEEMP_DIR="$NEEMP_DIR"

# ---------- 6. run benchmarks ----------
mkdir -p benchmarks/results
log "Running fast tests (sanity)…"
pytest -m "not slow" -q || fail "tests failed"

log "Running parallel speedup bench (set01 500 mol, all backends)…"
python examples/07_parallel_bench.py --set 01 --n-mols 500 --workers "$N_WORKERS" \
    | tee benchmarks/results/linux_parallel_set01_500.txt

log "Running DENewton on set03 17769 mol with parallel backend…"
python examples/06_scale_denewton_only.py --set 03 --n-mols 17769 \
    | tee benchmarks/results/linux_denewton_set03_17769.txt

log "All benches complete. Results in $WORK_DIR/benchmarks/results/"
