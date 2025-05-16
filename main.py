import os
import zipfile
import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from utils import plot_history, save_confusion_matrix, save_classification_report

# Set up logging to display INFO level messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
tf.get_logger().setLevel(logging.INFO)

def prepare_cats_vs_dogs(data_root: str = "cats_and_dogs_dataset") -> Tuple[str, str]:
    """
    Download and extract the Cats vs Dogs dataset, validating the structure.

    Args:
        data_root: Directory to extract dataset into.

    Returns:
        Tuple containing paths to training and validation directories.

    Raises:
        FileNotFoundError: If expected directories or subfolders are missing after extraction.
    """
    # Download dataset archive if not already present
    archive_path = tf.keras.utils.get_file(
        fname="cats_and_dogs_filtered.zip",
        origin="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
        extract=False
    )
    # Ensure the data_root directory exists
    os.makedirs(data_root, exist_ok=True)

    # Extract the downloaded zip file into data_root
    with zipfile.ZipFile(archive_path, 'r') as z:
        z.extractall(data_root)

    # Define train and validation directories
    base_dir = os.path.join(data_root, "cats_and_dogs_filtered")
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "validation")

    # Validate the structure
    for directory in (train_dir, val_dir):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        cats_dir = os.path.join(directory, "cats")
        dogs_dir = os.path.join(directory, "dogs")
        if not (os.path.isdir(cats_dir) and os.path.isdir(dogs_dir)):
            raise FileNotFoundError(f"Subfolders 'cats' and 'dogs' not found in {directory}")
        # Count images for logging
        cat_files = len(os.listdir(cats_dir))
        dog_files = len(os.listdir(dogs_dir))
        if cat_files == 0 or dog_files == 0:
            raise FileNotFoundError(f"No images found in {directory}: cats={cat_files}, dogs={dog_files}")
        logging.info(f"{directory}: {cat_files} cat images, {dog_files} dog images")

    logging.info(f"Train directory: {train_dir}")
    logging.info(f"Validation directory: {val_dir}")
    return train_dir, val_dir


def load_data(data_dir: str, img_size: Tuple[int, int] = (224, 224), batch_size: int = 32) -> tf.data.Dataset:
    """
    Load images from a directory into a tf.data.Dataset for binary classification.

    Args:
        data_dir: Directory with class subfolders (cats, dogs).
        img_size: Target image size for resizing.
        batch_size: Number of samples per batch.

    Returns:
        A tf.data.Dataset yielding (images, labels).
    """
    return tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'  # outputs 0 for cats, 1 for dogs
    )


def build_model() -> tf.keras.Model:
    """
    Build and compile a transfer-learning model using MobileNetV2.

    Returns:
        A compiled Keras Model ready for training.
    """
    # Load MobileNetV2 as feature extractor (without top classification layers)
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        pooling='avg',  # global average pooling at end
        input_shape=(224, 224, 3)
    )
    # Freeze the base to prevent its weights from updating
    base_model.trainable = False

    # Build custom top layers
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)  # preprocess inputs
    x = base_model(x, training=False)  # feature extraction
    x = tf.keras.layers.Dense(128, activation='relu')(x)  # classification head
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # binary output

    model = tf.keras.Model(inputs, outputs)
    # Compile with optimizer, loss, and accuracy metric
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    # Print model summary for verification
    model.summary()
    return model


def train_and_evaluate(model: tf.keras.Model,
                       train_ds: tf.data.Dataset,
                       val_ds: tf.data.Dataset,
                       epochs: int = 5) -> None:
    """
    Train the model and evaluate performance, saving plots and reports.

    Args:
        model: The compiled Keras model.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        epochs: Number of training epochs.
    """
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2
    )
    # Plot training history (accuracy/loss) and save figures
    plot_history(history, output_dir='plots')

    # Aggregate true labels from validation set
    y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
    # Predict probabilities on validation set
    y_pred_prob = model.predict(val_ds)
    # Convert probabilities to binary predictions (threshold 0.5)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Save text classification report and confusion matrix image
    save_classification_report(y_true, y_pred, output_path='plots/classification_report.txt')
    save_confusion_matrix(y_true, y_pred, output_path='plots/confusion_matrix.png')


def main() -> None:
    """
    Execute the full pipeline: data prep, training, evaluation, and save model.
    """
    # Prepare dataset directories
    train_dir, val_dir = prepare_cats_vs_dogs()
    # Load datasets
    train_ds = load_data(train_dir)
    val_ds = load_data(val_dir)
    # Build and train model
    model = build_model()
    train_and_evaluate(model, train_ds, val_ds, epochs=5)

    # Create directory for saved models
    os.makedirs('models', exist_ok=True)
    # Save the trained model for deployment
    model.save('models/cats_dogs_mobilenet_v2.h5')
    logging.info("Model saved to models/cats_dogs_mobilenet_v2.h5")

if __name__ == '__main__':
    main()