#!/bin/bash
# this script is for running on the cluster

set -e 

echo "Starting automated setup "

# Parse flags
KEEP_VENV=false
for arg in "$@"; do
    case "$arg" in
        --keep-venv)
            KEEP_VENV=true
            ;;
    esac
done

if [ ! -d "test_and_forge_cudakernels" ]; then
    echo "Cloning test_and_forge_cudakernels repository..."
    git clone -b topcdf git@github.com:sankeerth95/test_and_forge_cudakernels.git
else
    echo "test_and_forge_cudakernels directory already exists, skipping clone."
fi

if [ ! -d "vdit_runs_updated" ]; then
    echo "Cloning vdit_runs repository..."
    git clone -b topcdf git@github.com:sankeerth95/vdit_runs_updated.git
else
    echo "vdit_runs directory already exists, skipping clone."
fi

echo "Initializing submodules..."
cd test_and_forge_cudakernels
# Clean up any failed submodule attempts and cached data
rm -rf submodules/ThunderKittens
rm -rf .git/modules/submodules/ThunderKittens
# Update .gitmodules file to use SSH instead of HTTPS
git config --file=.gitmodules submodule.submodules/ThunderKittens.url git@github.com:sankeerth95/TK_fork.git
git submodule sync
git submodule update
cd ..


# ffmpeg is already available on the cluster, no need to install
# apt-get update -y
# apt-get install -y ffmpeg

# Load CUDA module (required for building CUDA extensions)
echo "Loading CUDA module..."
module load cuda/12.6

# Load OpenCV (and gcc) module before creating venv so it is visible inside
echo "Detecting and loading OpenCV module..."
# Try to pick the first available opencv/<version>
opencv_module=$(module avail opencv 2>&1 | awk '/opencv\//{print $1; exit}')
if [ -n "$opencv_module" ]; then
    module load gcc "$opencv_module"
    echo "Loaded OpenCV module: $opencv_module"
else
    echo "Warning: No OpenCV module found via 'module avail opencv'. Skipping load."
fi

# Create and use a dedicated Python virtual environment
VENV_DIR="$HOME/.venvs/tianlei"
echo "Setting up Python virtual environment at $VENV_DIR ..."
mkdir -p "$HOME/.venvs"
if [ ! -d "$VENV_DIR" ]; then
    # Include system site-packages so cluster modules (e.g., OpenCV) are visible in venv
    python3 -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
PY="$VENV_DIR/bin/python"
PIP="$PY -m pip"

# Upgrade build tooling
$PIP install --upgrade pip setuptools wheel

# Install PyTorch first (required for building CUDA extensions)
echo "Installing PyTorch..."
$PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# install CUDA extension
echo "Installing CUDA extension..."
cd test_and_forge_cudakernels/python



# Create pyproject.toml with torch as build dependency to fix offload_state_dict issue
echo "Creating pyproject.toml with torch as build dependency..."
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools", "wheel", "torch"]
build-backend = "setuptools.build_meta"
EOF

# Force clean recompilation by removing build artifacts
echo "Cleaning previous build artifacts..."
rm -rf build/ dist/ *.egg-info spattn.egg-info
rm -f spattn/*.so

# Force rebuild the CUDA extension
echo "Building CUDA extension (this may take a few minutes)..."
$PY setup.py build_ext --inplace --force

# Install in editable mode
echo "Installing in editable mode..."
$PY -m pip install -e . --no-build-isolation

cd ../..

# install required libraries for t2v_wan21.py
echo "Installing required libraries for t2v_wan21.py..."
$PIP install accelerate transformers mediapy
$PIP install "diffusers==0.33.0"
$PIP install -U ftfy
$PIP install -U sentencepiece regex

# install video export dependencies
echo "Installing video export dependencies..."
$PIP install imageio imageio-ffmpeg
$PIP install hf_transfer

# install NVIDIA nvCOMP Python bindings required by spattn.mask_utils
echo "Installing NVIDIA nvCOMP bindings..."
# Try cluster wheelhouse first; if missing, fall back to PyPI by unsetting
# environment variables that force Compute Canada's wheelhouse.
if ! $PIP install nvidia-nvcomp-cu12; then
    echo "Wheelhouse did not have nvidia-nvcomp-cu12; trying PyPI..."
    env -u PIP_NO_INDEX -u PIP_FIND_LINKS $PIP install --index-url https://pypi.org/simple nvidia-nvcomp-cu12 || echo "nvCOMP not available; skipping"
fi

echo "Running test script..."
cd vdit_runs_updated
# mkdir -p generated_vids
$PY t2v_wan21.py

echo "Setup and test completed"

# Optionally keep user in an interactive shell with venv active
if [ "$KEEP_VENV" = "true" ]; then
    echo "Opening interactive shell with virtualenv active (type 'exit' to leave)."
    echo "Venv path: $VENV_DIR"
    # Start a new bash with the venv already sourced
    exec bash --rcfile <(echo "source $VENV_DIR/bin/activate; PS1='(tianlei) \u@\h:\w\$ '")
fi