"""CNN model architecture for ASL sign classification."""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

import config


def build_cnn(num_classes: int = config.NUM_ASL_SIGNS,
              input_shape: tuple = config.INPUT_SHAPE) -> Sequential:
    """Build and compile the CNN classifier.

    Args:
        num_classes: Number of ASL sign classes.
        input_shape: Shape of input images (height, width, channels).

    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=input_shape, activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
