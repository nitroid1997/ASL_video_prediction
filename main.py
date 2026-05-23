#!/usr/bin/env python3
"""ASL Sign Language Predictor - Main Entry Point.

Usage:
    python main.py train  --train_dir data/train --test_dir data/test
    python main.py predict --image path/to/image.jpg
    python main.py live    [--webcam_index 0]
"""

import argparse
import json
import os
import sys

import config


def train(args):
    """Train the CNN model on the ASL dataset."""
    from asl_model.model import build_cnn
    from asl_model.data import create_data_generators

    train_dir = args.train_dir or config.TRAIN_DATA_DIR
    test_dir = args.test_dir or config.TEST_DATA_DIR

    if not os.path.isdir(train_dir):
        print(f"Error: Training directory not found: {train_dir}")
        sys.exit(1)
    if not os.path.isdir(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        sys.exit(1)

    epochs = args.epochs or config.EPOCHS
    batch_size = args.batch_size or config.BATCH_SIZE

    print(f"Training directory : {train_dir}")
    print(f"Test directory     : {test_dir}")
    print(f"Epochs             : {epochs}")
    print(f"Batch size         : {batch_size}")

    training_set, test_set = create_data_generators(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=batch_size,
    )

    num_classes = training_set.num_classes
    print(f"Detected {num_classes} classes: {list(training_set.class_indices.keys())}")

    classifier = build_cnn(num_classes=num_classes)
    classifier.summary()

    classifier.fit(
        training_set,
        steps_per_epoch=len(training_set),
        epochs=epochs,
        validation_data=test_set,
        validation_steps=len(test_set),
    )

    os.makedirs(config.MODEL_ASSETS_DIR, exist_ok=True)

    model_path = args.model_path or config.MODEL_PATH
    classifier.save(model_path)
    print(f"Model saved to: {model_path}")

    indices_path = args.class_indices_path or config.CLASS_INDICES_PATH
    with open(indices_path, "w") as f:
        json.dump(training_set.class_indices, f, indent=2)
    print(f"Class indices saved to: {indices_path}")


def predict(args):
    """Predict the ASL sign in a single image."""
    from asl_model.predict import predict_single_image

    model_path = args.model_path or config.MODEL_PATH
    indices_path = args.class_indices_path or config.CLASS_INDICES_PATH

    if not os.path.isfile(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Train a model first with: python main.py train")
        sys.exit(1)

    label = predict_single_image(
        image_path=args.image,
        model_path=model_path,
        class_indices_path=indices_path,
    )
    print(f"Predicted ASL sign: {label}")


def live(args):
    """Run live webcam ASL prediction."""
    from asl_model.predict import run_live_prediction

    model_path = args.model_path or config.MODEL_PATH
    indices_path = args.class_indices_path or config.CLASS_INDICES_PATH

    if not os.path.isfile(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Train a model first with: python main.py train")
        sys.exit(1)

    run_live_prediction(
        model_path=model_path,
        class_indices_path=indices_path,
        webcam_index=args.webcam_index,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="ASL Sign Language Predictor using CNN (Keras + OpenCV)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train the CNN model")
    train_parser.add_argument("--train_dir", type=str, default=None,
                              help="Path to training dataset directory")
    train_parser.add_argument("--test_dir", type=str, default=None,
                              help="Path to test/validation dataset directory")
    train_parser.add_argument("--epochs", type=int, default=None,
                              help=f"Number of training epochs (default: {config.EPOCHS})")
    train_parser.add_argument("--batch_size", type=int, default=None,
                              help=f"Batch size (default: {config.BATCH_SIZE})")
    train_parser.add_argument("--model_path", type=str, default=None,
                              help="Path to save the trained model")
    train_parser.add_argument("--class_indices_path", type=str, default=None,
                              help="Path to save the class indices JSON")

    # --- predict ---
    pred_parser = subparsers.add_parser("predict", help="Predict ASL sign from an image")
    pred_parser.add_argument("--image", type=str, required=True,
                             help="Path to the image file")
    pred_parser.add_argument("--model_path", type=str, default=None,
                             help="Path to the saved model")
    pred_parser.add_argument("--class_indices_path", type=str, default=None,
                             help="Path to class indices JSON")

    # --- live ---
    live_parser = subparsers.add_parser("live", help="Live webcam ASL prediction")
    live_parser.add_argument("--webcam_index", type=int,
                             default=config.DEFAULT_WEBCAM_INDEX,
                             help="Webcam device index (default: 0)")
    live_parser.add_argument("--model_path", type=str, default=None,
                             help="Path to the saved model")
    live_parser.add_argument("--class_indices_path", type=str, default=None,
                             help="Path to class indices JSON")

    return parser.parse_args()


def main():
    args = parse_args()

    commands = {
        "train": train,
        "predict": predict,
        "live": live,
    }

    if args.command not in commands:
        print("Please specify a command: train, predict, or live")
        print("Run 'python main.py --help' for usage information.")
        sys.exit(1)

    commands[args.command](args)


if __name__ == "__main__":
    main()
