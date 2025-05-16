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

    # Print dataset statistics for diagnosis
    num_train_cats = len(os.listdir(os.path.join(train_dir, "cats")))
    num_train_dogs = len(os.listdir(os.path.join(train_dir, "dogs")))
    num_val_cats = len(os.listdir(os.path.join(val_dir, "cats")))
    num_val_dogs = len(os.listdir(os.path.join(val_dir, "dogs")))
    
    logging.info(f"Dataset counts:")
    logging.info(f"Training: {num_train_cats} cats, {num_train_dogs} dogs")
    logging.info(f"Validation: {num_val_cats} cats, {num_val_dogs} dogs")
    
    return train_dir, val_dir

def load_data(data_dir: str,
              img_size: Tuple[int, int] = (224, 224),
              batch_size: int = 32,
              is_training: bool = True) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from images in a directory.
    Separates the preprocessing for training and validation.
    """
    # Changed label_mode to 'int' for proper integer class labels
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int',  # Use integer labels for proper classification
        seed=42,  # Set seed for reproducibility
        shuffle=is_training  # Only shuffle during training
    )
    
    # Apply preprocessing
    preprocess_func = tf.keras.applications.mobilenet_v2.preprocess_input
    ds = ds.map(lambda x, y: (preprocess_func(x), y), 
                    num_parallel_calls=tf.data.AUTOTUNE)
    
    # Enable prefetching for better performance
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def analyze_dataset(ds, name="Dataset"):
    """
    Analyze a dataset's characteristics outside of the graph.
    """
    logging.info(f"\n--- Analyzing {name} ---")
    
    # Get one batch
    for images, labels in ds.take(1):
        logging.info(f"Batch shape: X={images.shape}, y={labels.shape}")
        logging.info(f"Value range: [{tf.reduce_min(images).numpy()}, {tf.reduce_max(images).numpy()}]")
        
        # Check label distribution
        unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
        logging.info(f"Labels: {unique_labels}, Counts: {counts}")
        
        # Check for infinity or NaN values
        has_inf = tf.reduce_any(tf.math.is_inf(images)).numpy()
        has_nan = tf.reduce_any(tf.math.is_nan(images)).numpy()
        logging.info(f"Contains infinity: {has_inf}, Contains NaN: {has_nan}")
        
        # Sample a few images
        for i in range(min(3, images.shape[0])):
            sample_img = images[i].numpy()
            sample_label = labels[i].numpy()
            logging.info(f"Sample {i} - Label: {sample_label}")
            logging.info(f"  Min: {np.min(sample_img)}, Max: {np.max(sample_img)}")
            
        break


def build_model(include_augmentation=True, reduced_regularization=False):
    """Build the model with optional augmentation and regularization settings"""
    
    # Data augmentation for training only
    augmenter = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ], name="augmentation")

    # Base model
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", pooling="avg",
        input_shape=(224,224,3)
    )
    
    # Initially freeze all base model layers
    base_model.trainable = False

    inputs = layers.Input(shape=(224,224,3))
    
    # Apply augmentation only if requested
    if include_augmentation:
        x = augmenter(inputs)
    else:
        x = inputs
    
    x = base_model(x)
    
    # Set regularization and dropout based on params
    dropout_rate = 0.2 if reduced_regularization else 0.4
    reg_value = 1e-6 if reduced_regularization else 1e-5
    
    # Classifier head
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(reg_value))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def visualize_augmentations(dataset):
    """Create visualizations of augmented training samples"""
    import matplotlib.pyplot as plt
    
    # Get a batch of images
    for images, labels in dataset.take(1):
        # Select 5 samples
        sample_images = images[:5].numpy()
        sample_labels = labels[:5].numpy()
        
        # Create augmenter
        augmenter = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomBrightness(0.2),
            layers.RandomContrast(0.2),
        ])
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # For each sample
        for i in range(len(sample_images)):
            # Plot original
            plt.subplot(5, 5, i*5+1)
            # Convert from [-1,1] to [0,1] for display
            display_img = (sample_images[i] - np.min(sample_images[i])) / (np.max(sample_images[i]) - np.min(sample_images[i]))
            plt.imshow(display_img)
            plt.title(f"Original ({sample_labels[i]})")
            plt.axis("off")
            
            # Create 4 augmented versions
            img_tensor = tf.convert_to_tensor(sample_images[i:i+1])
            for j in range(4):
                aug_img = augmenter(img_tensor, training=True)
                aug_img_np = aug_img[0].numpy()
                
                plt.subplot(5, 5, i*5+j+2)
                # Normalize for display
                display_img = (aug_img_np - np.min(aug_img_np)) / (np.max(aug_img_np) - np.min(aug_img_np))
                plt.imshow(display_img)
                plt.title(f"Aug {j+1}")
                plt.axis("off")
        
        # Save
        os.makedirs('plots', exist_ok=True)
        plt.tight_layout()
        plt.savefig('plots/augmentation_samples.png')
        plt.close()
        logging.info(f"Saved augmentation visualization to plots/augmentation_samples.png")
        break


def compare_training_inference(model, dataset):
    """Compare model behavior in training vs inference mode"""
    logging.info("\n--- Comparing Training vs Inference Mode ---")
    
    # Get a batch
    for images, labels in dataset.take(1):
        # Use first 5 samples
        samples = images[:5]
        sample_labels = labels[:5].numpy()
        
        # Get predictions in training mode (dropout active)
        train_preds = model(samples, training=True).numpy().flatten()
        
        # Get predictions in inference mode (dropout inactive)
        infer_preds = model(samples, training=False).numpy().flatten()
        
        # Compare results
        for i in range(len(sample_labels)):
            logging.info(f"Sample {i} (label={sample_labels[i]}):")
            logging.info(f"  Training mode: {train_preds[i]:.4f}, Inference mode: {infer_preds[i]:.4f}")
            logging.info(f"  Difference: {abs(train_preds[i] - infer_preds[i]):.4f}")
        
        break


def evaluate_with_disabled_components(model, val_ds, train_ds):
    """
    Evaluate the model with different components disabled to isolate causes.
    """
    logging.info("\n--- Component-wise Evaluation ---")
    
    # Get some data
    for val_images, val_labels in val_ds.take(1):
        for train_images, train_labels in train_ds.take(1):
            # 1. Normal prediction (with dropout in inference mode)
            val_preds = model.predict(val_images, verbose=0)
            val_acc = np.mean(((val_preds > 0.5).astype(int).flatten() == val_labels.numpy()))
            logging.info(f"Normal validation accuracy: {val_acc:.4f}")
            
            train_preds = model.predict(train_images, verbose=0)
            train_acc = np.mean(((train_preds > 0.5).astype(int).flatten() == train_labels.numpy()))
            logging.info(f"Normal training accuracy: {train_acc:.4f}")
            
            # 2. Create modified models
            # Get the original model's config
            config = model.get_config()
            
            # Load weights from original model
            tmp_weights_path = 'tmp_weights.weights.h5'
            model.save_weights(tmp_weights_path)
            
            # Test with augmentation removed
            no_aug_model = build_model(include_augmentation=False)
            no_aug_model.load_weights(tmp_weights_path)
            
            train_preds_no_aug = no_aug_model.predict(train_images, verbose=0)
            train_acc_no_aug = np.mean(((train_preds_no_aug > 0.5).astype(int).flatten() == train_labels.numpy()))
            logging.info(f"Training accuracy without augmentation: {train_acc_no_aug:.4f}")
            
            # Test with reduced regularization
            less_reg_model = build_model(reduced_regularization=True)
            less_reg_model.load_weights(tmp_weights_path)
            
            train_preds_less_reg = less_reg_model.predict(train_images, verbose=0)
            train_acc_less_reg = np.mean(((train_preds_less_reg > 0.5).astype(int).flatten() == train_labels.numpy()))
            logging.info(f"Training accuracy with reduced regularization: {train_acc_less_reg:.4f}")
            
            # Clean up
            if os.path.exists(tmp_weights_path):
                os.remove(tmp_weights_path)
            
            break


class TrainInferCompare(tf.keras.callbacks.Callback):
    """Callback to compare training mode vs inference mode during training"""
    
    def __init__(self, train_ds, val_ds):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2 == 0:  # Every other epoch
            logging.info(f"\n--- Epoch {epoch} Training vs Inference Mode Comparison ---")
            
            # Get samples
            for train_images, train_labels in self.train_ds.take(1):
                train_samples = train_images[:3]
                train_labels = train_labels[:3].numpy()
                
                # Training mode predictions
                train_preds_training = self.model(train_samples, training=True).numpy().flatten()
                
                # Inference mode predictions
                train_preds_inference = self.model(train_samples, training=False).numpy().flatten()
                
                # Report
                for i in range(len(train_labels)):
                    logging.info(f"Training sample {i} (label={train_labels[i]}):")
                    logging.info(f"  Training mode: {train_preds_training[i]:.4f}")
                    logging.info(f"  Inference mode: {train_preds_inference[i]:.4f}")
                    logging.info(f"  Difference: {abs(train_preds_training[i] - train_preds_inference[i]):.4f}")
                break


def train_and_evaluate(model, train_ds, val_ds, epochs=10):
    """Training function with diagnostics"""
    # Analyze datasets
    analyze_dataset(train_ds, "Training Dataset")
    analyze_dataset(val_ds, "Validation Dataset")
    
    # Visualize augmentations
    visualize_augmentations(train_ds)
    
    # Compare training vs inference behavior
    compare_training_inference(model, train_ds)
    
    # Setup callbacks
    es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    rl = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    tc = TrainInferCompare(train_ds, val_ds)
    
    # Train the model
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, callbacks=[es, rl, tc], verbose=2
    )
    
    # Plot training history
    plot_history(history, output_dir='plots')
    
    # Evaluate with different components disabled
    evaluate_with_disabled_components(model, val_ds, train_ds)
    
    # Standard evaluation
    val_results = model.evaluate(val_ds, verbose=0)
    train_results = model.evaluate(train_ds, verbose=0)
    logging.info(f"\n--- Final Evaluation ---")
    logging.info(f"Training: Loss={train_results[0]:.4f}, Accuracy={train_results[1]:.4f}")
    logging.info(f"Validation: Loss={val_results[0]:.4f}, Accuracy={val_results[1]:.4f}")
    
    # Confusion matrix and classification report
    y_true = []
    y_pred = []
    
    for images, labels in val_ds:
        batch_preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend((batch_preds > 0.5).astype(int).flatten())
    
    os.makedirs('plots', exist_ok=True)
    save_classification_report(y_true, y_pred, output_path='plots/classification_report.txt')
    save_confusion_matrix(y_true, y_pred, output_path='plots/confusion_matrix.png')
    
    return model


def main() -> None:
    train_dir, val_dir = prepare_cats_vs_dogs()
    
    # Use consistent batch size for both datasets
    batch_size = 32
    
    # Load datasets with appropriate flags
    train_ds = load_data(train_dir, batch_size=batch_size, is_training=True)
    val_ds = load_data(val_dir, batch_size=batch_size, is_training=False)
    
    model = build_model()
    model = train_and_evaluate(model, train_ds, val_ds, epochs=10)

    os.makedirs('models', exist_ok=True)
    model.save('models/cats_dogs_mobilenet_v2_debug.h5')
    logging.info("Model saved to models/cats_dogs_mobilenet_v2_debug.h5")

if __name__ == '__main__':
    main()