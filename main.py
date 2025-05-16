import os
import zipfile
import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, regularizers, callbacks

from utils import plot_history, save_confusion_matrix, save_classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
tf.get_logger().setLevel(logging.INFO)

def prepare_cats_vs_dogs(data_root: str = "cats_and_dogs_dataset") -> Tuple[str, str]:
    """
    Download and extract the Cats vs Dogs filtered dataset.
    Returns:
        train_dir, val_dir
    """
    archive_path = tf.keras.utils.get_file(
        fname="cats_and_dogs_filtered.zip",
        origin="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
        extract=False
    )
    os.makedirs(data_root, exist_ok=True)
    with zipfile.ZipFile(archive_path, 'r') as z:
        z.extractall(data_root)

    base = os.path.join(data_root, "cats_and_dogs_filtered")
    train_dir = os.path.join(base, "train")
    val_dir   = os.path.join(base, "validation")

    logging.info(f"Train directory: {train_dir}")
    logging.info(f"Validation directory: {val_dir}")
    return train_dir, val_dir

def load_data(data_dir: str,
              img_size: Tuple[int, int] = (224, 224),
              batch_size: int = 32) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from images in a directory.
    """
    # Changed label_mode to 'int' for proper integer class labels
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int',  # Use integer labels for proper classification
        seed=42  # Set seed for reproducibility
    )
    # Apply preprocessing and normalization to both training and validation data
    preprocess_func = tf.keras.applications.mobilenet_v2.preprocess_input
    ds = ds.map(lambda x, y: (preprocess_func(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Enable prefetching for better performance
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model():
    # Data augmentation for training only
    augmenter = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),  # Increased rotation for more variety
        layers.RandomZoom(0.2),      # Increased zoom for more variety
        layers.RandomBrightness(0.2),  # Added brightness variation
        layers.RandomContrast(0.2),    # Added contrast variation
    ], name="augmentation")

    # Base model
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", pooling="avg",
        input_shape=(224,224,3)
    )
    
    # Initially freeze all base model layers
    base_model.trainable = False

    inputs = layers.Input(shape=(224,224,3))
    
    # Apply augmentation only during training
    x = augmenter(inputs)
    
    # Base model (preprocessed inputs are fed directly)
    x = base_model(x)
    
    # More robust classifier head
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-5))(x)  # More neurons, lighter regularization
    x = layers.BatchNormalization()(x)  # Added batch normalization
    x = layers.Dropout(0.4)(x)  # Slightly less dropout
    
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-5))(x)  # Added second dense layer
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    
    # Higher learning rate for initial training with frozen base
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),  # Increased from 1e-4
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_and_evaluate(model, train_ds, val_ds, epochs=10):
    # Two-phase training: 1) Train top layers, 2) Fine-tune upper layers
    
    # Phase 1: Train only the top layers
    logging.info("Phase 1: Training top layers with frozen base model")
    es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    rl = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    
    history1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=5, callbacks=[es, rl], verbose=2
    )
    
    # Phase 2: Unfreeze the top layers of the base model and train with a lower learning rate
    logging.info("Phase 2: Fine-tuning with unfrozen top layers")
    base_model = [layer for layer in model.layers if layer.name == 'mobilenetv2_1.00_224'][0]
    base_model.trainable = True
    
    # Freeze all layers except the top 10
    for layer in base_model.layers[:-10]:
        layer.trainable = False
        
    # Recompile with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),  # Much lower learning rate for fine-tuning
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    # Continue training with the same callbacks
    history2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs-5, callbacks=[es, rl], verbose=2
    )
    
    # Combine histories for plotting
    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history[key]
    
    combined_history = type('obj', (object,), {'history': combined_history})
    plot_history(combined_history, output_dir='plots')

    # Evaluate on validation set
    y_true = []
    y_pred_prob = []
    
    # Collect validation data batch by batch (more memory efficient)
    for images, labels in val_ds:
        y_true.extend(labels.numpy())
        batch_preds = model.predict(images, verbose=0)
        y_pred_prob.extend(batch_preds.flatten())
    
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    logging.info(f"Validation accuracy: {np.mean(y_true == y_pred):.4f}")

    os.makedirs('plots', exist_ok=True)
    save_classification_report(y_true, y_pred, output_path='plots/classification_report.txt')
    save_confusion_matrix(y_true, y_pred, output_path='plots/confusion_matrix.png')
    
    return model


def main() -> None:
    train_dir, val_dir = prepare_cats_vs_dogs()
    
    # Use consistent batch size for both datasets
    batch_size = 32
    
    train_ds = load_data(train_dir, batch_size=batch_size)
    val_ds = load_data(val_dir, batch_size=batch_size)
    
    model = build_model()
    model = train_and_evaluate(model, train_ds, val_ds, epochs=15)  # Extended max epochs

    os.makedirs('models', exist_ok=True)
    model.save('models/cats_dogs_mobilenet_v2_improved.keras')
    logging.info("Model saved to models/cats_dogs_mobilenet_v2_improved.keras")

if __name__ == '__main__':
    main()