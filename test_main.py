# test_main.py
import os
import tensorflow as tf
from main import prepare_cats_vs_dogs, load_data, build_model

def test_prepare_dirs_exist(tmp_path):
    train_dir, val_dir = prepare_cats_vs_dogs(data_root=str(tmp_path/"data"))
    # Both directories should exist
    assert os.path.isdir(train_dir), f"{train_dir} missing"
    assert os.path.isdir(val_dir),   f"{val_dir} missing"

def test_load_data_shapes(tmp_path):
    # Use prepare to get real dirs
    train_dir, val_dir = prepare_cats_vs_dogs(data_root=str(tmp_path/"data2"))
    train_ds = load_data(train_dir, batch_size=2)
    # Grab one batch
    for images, labels in train_ds.take(1):
        assert images.shape[1:] == (224,224,3)
        assert labels.shape[0] == images.shape[0]
        break

def test_model_compile_and_predict():
    model = build_model()
    # Should compile & run on dummy data
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    dummy = tf.random.uniform((1,224,224,3))
    out = model(dummy)
    assert out.shape == (1,1)

def test_model_save(tmp_path):
    model = build_model()
    path = tmp_path / "mdl.h5"
    model.save(str(path))
    assert path.exists()
