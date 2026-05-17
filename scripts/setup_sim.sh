#!/usr/bin/env bash
# Idempotent setup for the simulation stack: Libero + Kinetix.
#
# Run this AFTER `uv sync --extra ml --extra sim --extra jax --extra oxe`.
# `uv sync` prunes any package not declared in pyproject.toml, so the two
# editable third-party installs need to be re-applied whenever sync runs.
#
# Usage:
#     bash scripts/setup_sim.sh                  # uses ~/sim as the clone root
#     SIM_ROOT=/custom/path bash scripts/setup_sim.sh

set -euo pipefail

SIM_ROOT="${SIM_ROOT:-$HOME/sim}"
LIBERO_DIR="$SIM_ROOT/libero"
KINETIX_DIR="$SIM_ROOT/kinetix"

mkdir -p "$SIM_ROOT"

# --- Libero -------------------------------------------------------------
if [ ! -d "$LIBERO_DIR" ]; then
    echo "[setup_sim] cloning Libero into $LIBERO_DIR"
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
fi

# Libero's top-level package directory has no __init__.py; setuptools'
# find_packages() returns empty without it, which yields a no-op editable
# install. Drop one in.
if [ ! -f "$LIBERO_DIR/libero/__init__.py" ]; then
    echo "[setup_sim] adding namespace __init__.py to Libero clone"
    touch "$LIBERO_DIR/libero/__init__.py"
fi

# Libero's libero/libero/__init__.py runs an interactive `input()` prompt
# on first import to ask about the dataset folder. Pre-seed ~/.libero/config.yaml
# with the defaults that get_default_path_dict() would produce.
if [ ! -f "$HOME/.libero/config.yaml" ]; then
    echo "[setup_sim] seeding ~/.libero/config.yaml"
    mkdir -p "$HOME/.libero"
    LIB="$LIBERO_DIR/libero/libero"
    cat > "$HOME/.libero/config.yaml" <<YAML
benchmark_root: $LIB
bddl_files: $LIB/bddl_files
init_states: $LIB/init_files
datasets: $LIB/../datasets
assets: $LIB/assets
YAML
fi

# --- Kinetix ------------------------------------------------------------
if [ ! -d "$KINETIX_DIR" ]; then
    echo "[setup_sim] cloning Kinetix into $KINETIX_DIR"
    git clone --depth 1 https://github.com/FlairOx/Kinetix.git "$KINETIX_DIR"
fi

# --- Editable installs -------------------------------------------------
echo "[setup_sim] uv pip install -e libero kinetix"
uv pip install -e "$LIBERO_DIR" -e "$KINETIX_DIR"

echo "[setup_sim] done. Smoke check:"
uv run python - <<'PY'
from libero.libero import benchmark
import kinetix, jax, mujoco
import tensorflow_datasets as tfds
print("  libero  :", list(benchmark.get_benchmark_dict().keys()))
print("  kinetix :", kinetix.__name__)
print("  jax     :", jax.__version__, "devices=", jax.devices())
print("  mujoco  :", mujoco.__version__)
print("  tfds    :", tfds.__version__)
PY
