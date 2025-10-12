#!/bin/bash
set -e 

echo "Starting automated setup "

if [ ! -d "test_and_forge_cudakernels" ]; then
    echo "Cloning test_and_forge_cudakernels repository..."
    git clone https://github.com/sankeerth95/test_and_forge_cudakernels.git
else
    echo "test_and_forge_cudakernels directory already exists, skipping clone."
fi

if [ ! -d "vdit_runs_updated" ]; then
    echo "Cloning vdit_runs repository..."
    git clone https://github.com/sankeerth95/vdit_runs_updated.git
else
    echo "vdit_runs directory already exists, skipping clone."
fi

echo "Initializing submodules..."
cd test_and_forge_cudakernels
git submodule init
git submodule update
cd ..


# for mediapy
apt-get update -y
apt-get install -y ffmpeg

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
python setup.py build_ext --inplace --force

# Install in editable mode
echo "Installing in editable mode..."
python -m pip install -e . --no-build-isolation

cd ../..

# install required libraries for t2v_wan21_torch.py
echo "Installing required libraries for t2v_wan21_torch.py..."
pip install accelerate transformers mediapy
pip install "diffusers==0.35.1"
pip install -U ftfy
pip install -U sentencepiece regex

# install video export dependencies
echo "Installing video export dependencies..."
pip install imageio imageio-ffmpeg
pip install opencv-python
pip install hf_transfer

# install NVIDIA nvCOMP Python bindings required by spattn.mask_utils
echo "Installing NVIDIA nvCOMP bindings..."
pip install nvidia-nvcomp-cu12

echo "Running test script..."
cd vdit_runs_updated
# mkdir -p generated_vids
python t2v_wandistilled.py

echo "Setup and test completed"