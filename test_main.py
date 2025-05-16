import os
import numpy as np
import tensorflow as tf
import pytest

from main import prepare_cats_vs_dogs, load_data, build_model, train_and_evaluate
from utils import plot_history, save_confusion_matrix, save_classification_report
from app import load_model as app_load_model, predict_cat_dog

@pytest.fixture
def dummy_dirs(tmp_path):
    return str(tmp_path)

def test_prepare_and_load(dummy_dirs):
    train_dir, val_dir = prepare_cats_vs_dogs(data_root=dummy_dirs)
    assert os.path.isdir(train_dir)
    assert os.path.isdir(val_dir)

    ds = load_data(train_dir, img_size=(224, 224), batch_size=1)
    for images, labels in ds.take(1):
        assert images.shape == (1, 224, 224, 3)
        assert labels.shape == (1,)
        break

def test_model_build_and_train(tmp_path):
    model = build_model()
    # create tiny dataset
    x = tf.random.uniform((2, 224, 224, 3))
    y = tf.constant([0, 1], dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(1)
    history = model.fit(ds, epochs=1, verbose=0)

    # plot_history should not error
    plot_history(history, output_dir=str(tmp_path / "plots"))

    # metrics functions should not error
    save_classification_report(np.array([0,1]), np.array([0,1]), str(tmp_path / "report.txt"))
    save_confusion_matrix(np.array([0,1]), np.array([0,1]), str(tmp_path / "cm.png"))

def test_app_predict(monkeypatch, tmp_path):
    # dummy model that returns 0.7 for dog
    class DummyModel:
        def predict(self, arr):
            return np.array([[0.7]])

    dm = DummyModel()
    monkeypatch.setattr('app.load_model', lambda path: dm)

    # create white image
    img = np.ones((224, 224, 3), dtype=np.uint8) * 255
    result = predict_cat_dog(img, dm)
    assert set(result) == {'Cat', 'Dog'}
    assert pytest.approx(result['Dog'], rel=1e-6) == 0.7
    assert pytest.approx(result['Cat'], rel=1e-6) == 0.3
