"""Prediction utilities for single images and live webcam feed."""

import json
import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

import config


def load_class_indices(path: str = config.CLASS_INDICES_PATH) -> dict:
    """Load class-index mapping from a JSON file and return inverse map.

    Args:
        path: Path to the class_indices.json file.

    Returns:
        Dictionary mapping integer indices to class label strings.
    """
    with open(path, "r") as f:
        label_map = json.load(f)
    return {v: k for k, v in label_map.items()}


def predict_single_image(
    image_path: str,
    model_path: str = config.MODEL_PATH,
    class_indices_path: str = config.CLASS_INDICES_PATH,
) -> str:
    """Predict the ASL sign in a single image.

    Args:
        image_path: Path to the image file.
        model_path: Path to the saved Keras model.
        class_indices_path: Path to the class indices JSON.

    Returns:
        Predicted class label string.
    """
    classifier = load_model(model_path)
    inverse_label_map = load_class_indices(class_indices_path)

    test_image = keras_image.load_img(image_path, target_size=config.IMAGE_SIZE)
    test_image = keras_image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0

    predictions = classifier.predict(test_image)
    predicted_class_index = int(np.argmax(predictions[0]))
    return inverse_label_map.get(predicted_class_index, "Unknown")


def run_live_prediction(
    model_path: str = config.MODEL_PATH,
    class_indices_path: str = config.CLASS_INDICES_PATH,
    webcam_index: int = config.DEFAULT_WEBCAM_INDEX,
) -> None:
    """Run live ASL prediction using a webcam feed.

    Args:
        model_path: Path to the saved Keras model.
        class_indices_path: Path to the class indices JSON.
        webcam_index: Index of the webcam device.
    """
    try:
        classifier = load_model(model_path)
        print(f"Model loaded from: {model_path}")
    except Exception as exc:
        print(f"Error loading model from {model_path}: {exc}")
        sys.exit(1)

    inverse_label_map = load_class_indices(class_indices_path)
    print(f"Detected classes: {list(inverse_label_map.values())}")

    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam (index={webcam_index}).")
        sys.exit(1)

    print("Webcam opened. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting.")
                break

            resized = cv2.resize(frame, config.IMAGE_SIZE)
            processed = keras_image.img_to_array(resized)
            processed = np.expand_dims(processed, axis=0) / 255.0

            predictions = classifier.predict(processed, verbose=0)
            idx = int(np.argmax(predictions[0]))
            label = inverse_label_map.get(idx, "Unknown")

            cv2.putText(
                frame, f"Prediction: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.imshow("ASL Live Prediction", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
