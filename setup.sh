#!/usr/bin/env bash
# ============================================================================
# ASL Sign Language Predictor - Dependency Check & Installation Script
# ============================================================================
set -e

REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=8

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------- Python version check ----------
check_python() {
    if ! command -v python3 &>/dev/null; then
        error "python3 is not installed. Please install Python ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}+."
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -lt "$REQUIRED_PYTHON_MAJOR" ] || \
       { [ "$PYTHON_MAJOR" -eq "$REQUIRED_PYTHON_MAJOR" ] && [ "$PYTHON_MINOR" -lt "$REQUIRED_PYTHON_MINOR" ]; }; then
        error "Python ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}+ is required (found $PYTHON_VERSION)."
        exit 1
    fi
    info "Python $PYTHON_VERSION detected."
}

# ---------- pip check ----------
check_pip() {
    if ! python3 -m pip --version &>/dev/null; then
        warn "pip not found. Attempting to install..."
        python3 -m ensurepip --upgrade || {
            error "Could not install pip. Please install it manually."
            exit 1
        }
    fi
    info "pip is available."
}

# ---------- Virtual environment ----------
setup_venv() {
    VENV_DIR=".venv"
    if [ ! -d "$VENV_DIR" ]; then
        info "Creating virtual environment in $VENV_DIR ..."
        python3 -m venv "$VENV_DIR"
    fi

    info "Activating virtual environment..."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    info "Virtual environment active: $(which python3)"
}

# ---------- Install dependencies ----------
install_deps() {
    info "Upgrading pip..."
    python3 -m pip install --upgrade pip

    info "Installing project dependencies..."
    python3 -m pip install -r requirements.txt
    info "All dependencies installed successfully."
}

# ---------- Verify imports ----------
verify_imports() {
    info "Verifying critical imports..."
    python3 -c "
import tensorflow as tf
import keras
import cv2
import numpy as np
print(f'  TensorFlow : {tf.__version__}')
print(f'  Keras      : {keras.__version__}')
print(f'  OpenCV     : {cv2.__version__}')
print(f'  NumPy      : {np.__version__}')
"
    info "All imports verified."
}

# ---------- Data directory scaffold ----------
scaffold_data() {
    if [ ! -d "data/train" ] || [ ! -d "data/test" ]; then
        warn "Dataset directories not found. Creating placeholder structure..."
        mkdir -p data/train data/test
        info "Created data/train and data/test directories."
        info "Place your ASL sign images in class-named sub-folders, e.g.:"
        info "  data/train/A/  data/train/B/  ...  data/train/Z/"
        info "  data/test/A/   data/test/B/   ...  data/test/Z/"
    else
        info "Dataset directories already exist."
    fi
}

# ---------- Main ----------
main() {
    echo "============================================"
    echo " ASL Sign Language Predictor - Setup"
    echo "============================================"
    echo

    check_python
    check_pip
    setup_venv
    install_deps
    verify_imports
    scaffold_data

    echo
    info "Setup complete! You can now run:"
    info "  source .venv/bin/activate"
    info "  python main.py train  --train_dir data/train --test_dir data/test"
    info "  python main.py predict --image path/to/image.jpg"
    info "  python main.py live"
}

main "$@"
