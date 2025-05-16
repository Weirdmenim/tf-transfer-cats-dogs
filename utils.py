import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def plot_history(history, output_dir: str = 'plots') -> None:
    """
    Plot and save training/validation accuracy and loss over epochs.
    Handles missing validation metrics gracefully.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get('accuracy', []), label='Train Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_accuracy.png'))
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get('loss', []), label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()

def save_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          output_path: str) -> None:
    """
    Generate and save a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_classification_report(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               output_path: str) -> None:
    """
    Save a text classification report including precision, recall, F1-score.
    """
    report = classification_report(y_true, y_pred, target_names=['Cat', 'Dog'])
    with open(output_path, 'w') as f:
        f.write(report)
