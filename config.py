"""Centralized configuration for the ASL Sign Language Predictor."""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ASSETS_DIR = os.path.join(BASE_DIR, "asl_model_assets")
MODEL_PATH = os.path.join(MODEL_ASSETS_DIR, "asl_classifier_model.h5")
CLASS_INDICES_PATH = os.path.join(MODEL_ASSETS_DIR, "class_indices.json")

# Default dataset paths (override via CLI arguments)
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DATA_DIR = os.path.join(BASE_DIR, "data", "test")

# --- Model Hyperparameters ---
NUM_ASL_SIGNS = 26
IMAGE_SIZE = (64, 64)
INPUT_SHAPE = (64, 64, 3)
BATCH_SIZE = 32
EPOCHS = 15

# --- Data Augmentation ---
RESCALE = 1.0 / 255
SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True

# --- Webcam ---
DEFAULT_WEBCAM_INDEX = 0
