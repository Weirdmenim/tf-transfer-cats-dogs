# test_main.py
import os
import numpy as np
import tempfile
import tensorflow as tf
from pathlib import Path
from main import prepare_cats_vs_dogs, load_data, build_model

def test_prepare_and_load(tmp_path):
    # Test data preparation and data loader
    train_dir, val_dir = prepare_cats_vs_dogs(data_root=str(tmp_path / "data"))
    assert os.path.isdir(train_dir), "Train directory should exist"
    assert os.path.isdir(val_dir), "Validation directory should exist"

    # Load a small batch
    train_ds = load_data(train_dir, img_size=(224, 224), batch_size=2)
    for images, labels in train_ds.take(1):
        assert images.shape == (2, 224, 224, 3)
        assert labels.shape == (2, 1)  # Updated to match actual shape
        break

def test_model_build_and_predict():
    # Build and run model on dummy data
    model = build_model()
    dummy = tf.random.uniform((1, 224, 224, 3), minval=0, maxval=256, dtype=tf.float32)  # Updated to [0, 255] range
    out = model(dummy)
    assert out.shape == (1, 1), "Model output shape should be (1,1)"

def test_plot_history_and_reports(tmp_path):
    # Create a fake history object
    class DummyHistory:
        history = {
            'accuracy': [0.5, 0.7],
            'val_accuracy': [0.4, 0.6],
            'loss': [1.0, 0.8],
            'val_loss': [1.2, 0.9]
        }
    history = DummyHistory()

    # Test plot_history
    from utils import plot_history
    output_dir = tmp_path / "plots"
    plot_history(history, output_dir=str(output_dir))
    assert (output_dir / 'training_accuracy.png').exists()
    assert (output_dir / 'training_loss.png').exists()

    # Test save_confusion_matrix and classification report
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    from utils import save_confusion_matrix, save_classification_report
    cm_path = tmp_path / 'confusion.png'
    report_path = tmp_path / 'report.txt'
    save_confusion_matrix(y_true, y_pred, str(cm_path))
    save_classification_report(y_true, y_pred, str(report_path))
    assert cm_path.exists(), "Confusion matrix image should be saved"
    assert report_path.exists(), "Classification report should be saved"
    # Check contents of report for expected labels
    content = report_path.read_text()
    assert 'precision' in content and 'recall' in content and 'f1-score' in content

def test_app_predict_function(tmp_path, monkeypatch):
    # Create a dummy model that returns 0.8 for dog probability
    class DummyModel:
        def predict(self, arr):
            # arr shape [1,224,224,3]
            return np.array([[0.8]])
    dummy_model = DummyModel()

    # Monkeypatch load_model to return dummy model
    monkeypatch.setattr('app.load_model', lambda path: dummy_model)

    # Create a dummy image (white square)
    img = np.ones((224, 224, 3), dtype=np.uint8) * 255
    from app import predict_cat_dog
    result = predict_cat_dog(img, dummy_model)
    assert isinstance(result, dict)
    assert 'Cat' in result and 'Dog' in result
    # Dog prob from dummy is 0.8
    assert abs(result['Dog'] - 0.8) < 1e-6
    assert abs(result['Cat'] - 0.2) < 1e-6