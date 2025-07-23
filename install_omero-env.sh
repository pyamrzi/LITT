#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Setting up Python 3.11 virtual environment for OMERO dependencies..."

# Update system packages
echo "Updating system packages..."
sudo apt-get update

# Install system dependencies first
echo "Installing system dependencies..."
sudo apt-get install -y \
    fish \
    nano \
    vim \
    build-essential \
    software-properties-common \
    python3-full \
    python3-pip

# Add deadsnakes PPA to get Python 3.11 on Ubuntu 24.04
echo "Adding deadsnakes PPA for Python 3.11..."
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update

# Install Python 3.11 and related packages
echo "Installing Python 3.11..."
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3.11-distutils

# Verify Python 3.11 installation
echo "Verifying Python 3.11 installation..."
python3.11 --version

# Create virtual environment with Python 3.11
VENV_NAME="omero-env"
echo "Creating virtual environment '$VENV_NAME' with Python 3.11..."

# Remove existing venv if it exists
if [ -d "$VENV_NAME" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_NAME"
fi

# Create new venv with Python 3.11
python3.11 -m venv "$VENV_NAME"

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Verify Python version in venv
echo "Python version in virtual environment:"
python --version

# Upgrade pip, setuptools, and wheel in the virtual environment
echo "Upgrading pip, setuptools, and wheel in virtual environment..."
python -m pip install --upgrade pip setuptools wheel

# Install Python packages in the virtual environment
echo "Installing Python packages..."

# URLs for specific wheels
ZEROC_ICE_WHL="https://github.com/glencoesoftware/zeroc-ice-py-linux-x86_64/releases/download/20240202/zeroc_ice-3.6.5-cp311-cp311-manylinux_2_28_x86_64.whl"
LAVLAB_UTILS_WHL="https://github.com/laviolette-lab/lavlab-python-utils/releases/download/v1.3.1/lavlab_python_utils-1.3.1-py3-none-any.whl"

# Install system dependencies for OpenCV and other libraries
echo "Installing additional system dependencies..."
sudo apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0t64 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libvips-dev

# Install packages in order (some may have dependencies on others)
echo "Installing core Python packages..."
python -m pip install \
    ipykernel \
    numpy==1.26 \
    matplotlib \
    tqdm \
    pillow==9.4.0 \
    shapely \
    ipynb \
    rasterio \
    ipywidgets \
    scipy \
    getpass4

# Install OpenCV
echo "Installing OpenCV..."
python -m pip install opencv-python

# Install scikit-image with compatible version for Python 3.11
echo "Installing scikit-image..."
python -m pip install scikit-image==0.21.0

# Install tifffile for TIFF image handling
echo "Installing tifffile..."
python -m pip install tifffile

# Install specific wheels from URLs
echo "Installing zeroc-ice wheel..."
python -m pip install "$ZEROC_ICE_WHL"

echo "Installing lavlab-python-utils wheel..."
python -m pip install "$LAVLAB_UTILS_WHL"

# Install OMERO-py
echo "Installing omero-py..."
python -m pip install omero-py

# Install VALIS for image registration
echo "Installing VALIS..."
python -m pip install valis-wsi==1.1.0

# Install pyvips
echo "Installing pyvips..."
python -m pip install pyvips

echo ""
echo "✅ Installation complete!"
echo ""
echo "⚠️  IMPORTANT NOTES:"
echo "1. There's a typo in your notebook LITT_WSI.ipynb on line 51:"
echo "   Change: from skimage.transform import resizels"
echo "   To:     from skimage.transform import resize"
echo ""
echo "2. To activate your virtual environment in the future, run:"
echo "    source $VENV_NAME/bin/activate"
echo ""
echo "3. To deactivate the virtual environment, run:"
echo "    deactivate"
echo ""
echo "Your virtual environment is now ready with Python $(python --version | cut -d' ' -f2)"
echo ""
echo "Test your installation by running:"
echo "    source $VENV_NAME/bin/activate"
echo "    python -c 'import matplotlib, numpy, omero, lavlab, cv2, tifffile, valis, pyvips; print(\"All imports successful!\")'"