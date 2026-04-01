#!/usr/bin/env bash
# AF3Score installation script
#
# Usage:
#   bash install.sh                          # install into the currently active conda env
#   bash install.sh --create-env af3score    # create a fresh conda env named 'af3score' and install into it
#   bash install.sh --pin-numpy              # pin numpy to 1.26.3 after install
#                                            #   (use when sharing env with caliby/pymol)
#   bash install.sh --create-env af3score --pin-numpy
#
# Prerequisites:
#   - conda is available
#   - CUDA 12.x is available on the system
#   - Run from the repo root: cd /path/to/af3score && bash install.sh

set -euo pipefail

PIN_NUMPY=0
CREATE_ENV=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --pin-numpy)   PIN_NUMPY=1; shift ;;
    --create-env)  CREATE_ENV="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Create conda env if requested
# ---------------------------------------------------------------------------
if [[ -n "$CREATE_ENV" ]]; then
  if conda env list | grep -qE "^${CREATE_ENV}\s"; then
    echo "==> Conda env '${CREATE_ENV}' already exists, using it."
  else
    echo "==> Creating conda env '${CREATE_ENV}' with Python 3.12..."
    conda create -y -n "$CREATE_ENV" python=3.12
  fi

  echo "==> Re-running installer inside '${CREATE_ENV}'..."
  EXTRA_FLAGS=""
  [[ "$PIN_NUMPY" -eq 1 ]] && EXTRA_FLAGS="--pin-numpy"
  conda run -n "$CREATE_ENV" --no-capture-output \
    bash "$REPO_DIR/install.sh" $EXTRA_FLAGS
  exit 0
fi

# ---------------------------------------------------------------------------
# From here on we are running inside the target env
# ---------------------------------------------------------------------------
PYTHON=$(python --version 2>&1)
echo "==> Python: $PYTHON"
echo "==> Conda env: ${CONDA_DEFAULT_ENV:-<none>}"
echo "==> Repo: $REPO_DIR"
echo ""

cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# 1. Build tools (pybind11 needed to compile C++ extensions)
# ---------------------------------------------------------------------------
echo "==> [1/5] Installing build tools..."
pip install "pybind11" "scikit-build-core"

# ---------------------------------------------------------------------------
# 2. JAX 0.9.1 with CUDA support
# ---------------------------------------------------------------------------
echo ""
echo "==> [2/5] Installing JAX 0.9.1..."
pip install "jax==0.9.1" "jaxlib==0.9.1"

# Install CUDA plugin with --no-deps so pip does not overwrite CUDA/cuDNN
# packages that a co-installed torch may depend on.
pip install \
  "jax-cuda12-plugin==0.9.1" \
  "jax-cuda12-pjrt==0.9.1" \
  --no-deps

# jax-cuda12-plugin 0.9.1 is compiled against cuDNN 9.8.
# cuDNN is forward-compatible within major version 9, so torch built against
# 9.1 will still run correctly once 9.8 is installed.
echo "==> Upgrading nvidia-cudnn-cu12 to >=9.8 (required by jax-cuda12-plugin 0.9.1)..."
pip install "nvidia-cudnn-cu12>=9.8"

# ---------------------------------------------------------------------------
# 3. Python dependencies
# ---------------------------------------------------------------------------
echo ""
echo "==> [3/5] Installing Python dependencies..."
pip install \
  "absl-py>=2.3.1" \
  "dm-haiku==0.0.16" \
  "dm-tree" \
  "h5py" \
  "tokamax==0.0.11" \
  "tqdm" \
  "zstandard" \
  "pandas" \
  "biopython"

# rdkit: skip if already installed (conda envs often have it pre-installed)
python -c "import rdkit" 2>/dev/null \
  || pip install "rdkit"

# ---------------------------------------------------------------------------
# 4. Build and install the alphafold3 package (compiles C++ extensions)
# ---------------------------------------------------------------------------
echo ""
echo "==> [4/5] Building and installing alphafold3 package..."
pip install --no-deps -e .

# ---------------------------------------------------------------------------
# 5. Build CCD / chemical component data assets
# ---------------------------------------------------------------------------
echo ""
echo "==> [5/5] Building data assets (CCD)..."
build_data

# ---------------------------------------------------------------------------
# Optional: pin numpy for caliby/pymol compatibility
# ---------------------------------------------------------------------------
if [[ "$PIN_NUMPY" -eq 1 ]]; then
  echo ""
  echo "==> Pinning numpy to 1.26.3 (--pin-numpy)..."
  pip install "numpy==1.26.3"
fi

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
echo ""
echo "==> Smoke test..."
python - <<'EOF'
import jax, jax.numpy as jnp, haiku as hk, tokamax
from alphafold3.model import model, params
from alphafold3.constants import chemical_components

devices = jax.devices()
print(f"  jax {jax.__version__}  |  devices: {devices}")
x = jnp.ones((4, 4))
print(f"  jax compute: {jnp.dot(x, x)[0,0]}")
print("  alphafold3 imports OK")
EOF

echo ""
echo "==> Installation complete."
echo ""
echo "    Pipeline steps:"
echo "      1_extract_chains.py  --input_dir <pdb_dir> --output_dir <out>"
echo "      2_pdb2jax.py         --pdb_dir <pdb_dir>   --output_dir <out>"
echo "      3_generate_json.py   --sequence_csv <csv>  --cif_dir <cif> --output_dir <out>"
echo "      run_af3score.py      --batch_json_dir <jsons> --batch_h5_dir <h5s> --output_dir <out> --model_dir <weights_dir>"
echo "      04_get_metrics.py    --pdb_dir <pdb_dir>   --af3_output_dir <out> --output_csv metrics.csv"
echo ""
echo "    Or use the orchestrator:"
echo "      af3score_pipeline.py --input_dir <pdb_dir> --output_dir <out> --weights <path/to/weights.npz>"
