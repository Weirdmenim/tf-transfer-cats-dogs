import os
import zipfile
import logging

import tensorflow as tf

# Configure logging
tf.get_logger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from utils import plot_history

def prepare_cats_vs_dogs(data_root="cats_and_dogs_dataset"):
    """
    Download and extract the Cats vs Dogs filtered dataset.

    Args:
        data_root (str): Directory to extract the dataset into.

    Returns:
        Tuple[str, str]: Paths to training and validation directories.
    """
    # Download the zip archive
    archive_path = tf.keras.utils.get_file(
        fname="cats_and_dogs_filtered.zip",
        origin="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
        extract=False
    )
    # Ensure data_root
    os.makedirs(data_root, exist_ok=True)
    # Extract archive into data_root
    with zipfile.ZipFile(archive_path, 'r') as z:
        z.extractall(data_root)
    # Define directories
    base = os.path.join(data_root, "cats_and_dogs_filtered")
    train_dir = os.path.join(base, "train")
    val_dir   = os.path.join(base, "validation")
    logging.info(f"Train dir: {train_dir}")
    logging.info(f"Validation dir: {val_dir}")
    return train_dir, val_dir


def load_data(data_dir, img_size=(224, 224), batch_size=32):
    """
    Create a tf.data.Dataset from images in a directory.
    """
    return tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )


def build_model():
    """
    Build a transfer-learning model using MobileNetV2 from tf.keras.applications.
    """
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        pooling='avg',
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_and_plot(model, train_ds, val_ds, epochs=5):
    """
    Train model and plot results.
    """
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2
    )
    plot_history(history)
    return history


def run_inference(model, img_path, img_size=(224, 224)):
    """
    Run inference on a single image.
    """
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = tf.expand_dims(x, 0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    pred = float(model.predict(x)[0][0])
    label = 'Dog' if pred > 0.5 else 'Cat'
    logging.info(f"Inference on {os.path.basename(img_path)}: {label} ({pred:.2f})")


def main():
    train_dir, val_dir = prepare_cats_vs_dogs()
    train_ds = load_data(train_dir)
    val_ds   = load_data(val_dir)
    model    = build_model()

    train_and_plot(model, train_ds, val_ds, epochs=5)
    model.save('cats_dogs_mobilenet_v2.h5')

    # Run inference on a sample
    sample_cat = tf.io.gfile.glob(os.path.join(train_dir, 'cats', '*.jpg'))[0]
    run_inference(model, sample_cat)


if __name__ == "__main__":
    main()
