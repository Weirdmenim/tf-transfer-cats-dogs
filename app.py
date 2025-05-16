# app.py
import os
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image


def load_model(model_path: str) -> tf.keras.Model:
    """
    Load and return the trained model from the given path.

    Args:
        model_path: Path to the saved Keras model file.

    Returns:
        Loaded TensorFlow Keras model.
    """
    return tf.keras.models.load_model(model_path)


def predict_cat_dog(image, model: tf.keras.Model) -> dict:
    """
    Preprocess an uploaded image and generate classification probabilities.

    Args:
        image: Uploaded image (numpy array).
        model: Loaded Keras model.

    Returns:
        Dictionary with class probabilities for 'Cat' and 'Dog'.
    """
    if image is None:
        return {'Cat': 0.0, 'Dog': 0.0}

    # Convert numpy array to PIL Image for processing
    img = Image.fromarray(image).convert('RGB')
    # Resize image to model input size
    img = img.resize((224, 224))
    # Normalize pixel values to [0,1]
    arr = np.array(img) / 255.0
    # Add batch dimension
    arr = arr.reshape(1, 224, 224, 3)

    # Predict probability of 'Dog' class
    pred_prob = float(model.predict(arr)[0][0])
    # Return probabilities for both classes
    return {'Cat': 1 - pred_prob, 'Dog': pred_prob}


def main() -> None:
    """
    Launch the Gradio web interface for Cat vs Dog classification.
    """
    # Build model path
    model_path = os.path.join('models', 'cats_dogs_mobilenet_v2.h5')
    # Load the model once at startup
    model = load_model(model_path)

    # Define Gradio interface
    interface = gr.Interface(
        fn=lambda img: predict_cat_dog(img, model),
        inputs=gr.Image(type="numpy", label="Upload an image of a cat or dog"),
        outputs=gr.Label(classes=['Cat', 'Dog'], num_top_classes=2),
        title='🐱🐶 Cats vs Dogs Classifier',
        description='Upload an image file of a cat or dog to receive classification probabilities.'
    )
    # Launch the app
    interface.launch()

if __name__ == '__main__':
    main()
