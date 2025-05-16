# utils.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def plot_history(history, output_dir: str = 'plots') -> None:
    """
    Plot and save training/validation accuracy and loss over epochs.

    Args:
        history: Keras History object returned by model.fit().
        output_dir: Directory to save plot images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_accuracy.png'))
    plt.close()

    # Plot loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str) -> None:
    """
    Generate and save a confusion matrix heatmap.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        output_path: File path to save the image.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.colorbar()
    # Annotate matrix cells with counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, output_path: str) -> None:
    """
    Save a text classification report including precision, recall, F1-score.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        output_path: File path to save the report.
    """
    report = classification_report(y_true, y_pred, target_names=['Cat', 'Dog'])
    with open(output_path, 'w') as f:
        f.write(report)
