# ASL Video Prediction

A CNN-based American Sign Language (ASL) sign classifier built with **Keras** (TensorFlow backend) and **OpenCV**. Supports training on custom datasets, single-image prediction, and real-time webcam prediction.

## Project Structure

```
ASL_video_prediction/
├── main.py                 # CLI entry point (train / predict / live)
├── config.py               # Centralized configuration & hyperparameters
├── asl_model/
│   ├── __init__.py
│   ├── model.py            # CNN architecture (Sequential)
│   ├── data.py             # Data loading & augmentation
│   └── predict.py          # Single-image & live-webcam prediction
├── requirements.txt        # Python dependencies
├── setup.sh                # Dependency check & installation script
├── notebooks/
│   └── AvsB.ipynb          # Original Jupyter notebook
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
chmod +x setup.sh
./setup.sh
```

The script will:
- Verify Python 3.8+ and pip are available
- Create a virtual environment (`.venv/`)
- Install all dependencies from `requirements.txt`
- Verify that TensorFlow, Keras, OpenCV, and NumPy import correctly
- Scaffold `data/train` and `data/test` placeholder directories

### 2. Prepare Your Dataset

Organize your ASL images into class-named sub-folders:

```
data/
├── train/
│   ├── A/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── B/
│   └── ...
└── test/
    ├── A/
    ├── B/
    └── ...
```

### 3. Train the Model

```bash
source .venv/bin/activate
python main.py train --train_dir data/train --test_dir data/test
```

Optional flags:
| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 15 | Number of training epochs |
| `--batch_size` | 32 | Training batch size |
| `--model_path` | `asl_model_assets/asl_classifier_model.h5` | Where to save the model |
| `--class_indices_path` | `asl_model_assets/class_indices.json` | Where to save class mapping |

### 4. Predict a Single Image

```bash
python main.py predict --image path/to/asl_sign.jpg
```

### 5. Live Webcam Prediction

```bash
python main.py live [--webcam_index 0]
```

Press **q** to quit the video stream.

## Configuration

All default values (image size, batch size, epochs, paths, augmentation params) are defined in [`config.py`](config.py). You can edit them directly or override via CLI arguments.

## Model Architecture

| Layer | Details |
|-------|---------|
| Conv2D | 32 filters, 3x3, ReLU |
| MaxPooling2D | 2x2 |
| Conv2D | 32 filters, 3x3, ReLU |
| MaxPooling2D | 2x2 |
| Flatten | — |
| Dense | 128 units, ReLU |
| Dense (output) | *N* units (one per class), Softmax |

- **Optimizer:** Adam
- **Loss:** Categorical Cross-Entropy
- **Input size:** 64 x 64 x 3

## Requirements

- Python 3.8+
- TensorFlow >= 2.10
- Keras >= 2.10
- OpenCV >= 4.6
- NumPy >= 1.23

## License

MIT License - see [LICENSE](LICENSE) for details.
