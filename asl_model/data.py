"""Data loading and augmentation utilities."""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config


def create_data_generators(
    train_dir: str = config.TRAIN_DATA_DIR,
    test_dir: str = config.TEST_DATA_DIR,
    image_size: tuple = config.IMAGE_SIZE,
    batch_size: int = config.BATCH_SIZE,
):
    """Create training and test data generators with augmentation.

    Args:
        train_dir: Path to the training dataset directory.
        test_dir: Path to the test/validation dataset directory.
        image_size: Target size for resizing images.
        batch_size: Number of images per batch.

    Returns:
        Tuple of (training_generator, test_generator).
    """
    train_datagen = ImageDataGenerator(
        rescale=config.RESCALE,
        shear_range=config.SHEAR_RANGE,
        zoom_range=config.ZOOM_RANGE,
        horizontal_flip=config.HORIZONTAL_FLIP,
    )

    test_datagen = ImageDataGenerator(rescale=config.RESCALE)

    training_set = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
    )

    test_set = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
    )

    return training_set, test_set
