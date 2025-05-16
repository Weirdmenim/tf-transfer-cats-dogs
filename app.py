import os
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

def load_model(model_path: str) -> tf.keras.Model:
    """
    Load and return the trained model from disk.
    """
    return tf.keras.models.load_model(model_path)

def predict_cat_dog(image, model: tf.keras.Model) -> dict:
    """
    Preprocess an uploaded image and return classification probabilities.
    """
    if image is None:
        return {'Cat': 0.0, 'Dog': 0.0}

    img = Image.fromarray(image).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = arr.reshape(1, 224, 224, 3)

    pred_prob = float(model.predict(arr)[0][0])
    return {'Cat': 1 - pred_prob, 'Dog': pred_prob}

def main() -> None:
    """
    Launch the Gradio interface.
    """
    model = load_model(os.path.join('models', 'cats_dogs_mobilenet_v2.keras'))

    interface = gr.Interface(
        fn=lambda img: predict_cat_dog(img, model),
        inputs=gr.Image(type="numpy", label="Upload a cat or dog image"),
        outputs=gr.Label(classes=['Cat', 'Dog'], num_top_classes=2),
        title='🐱🐶 Cats vs Dogs Classifier',
        description='Upload an image of a cat or dog to classify.'
    )
    interface.launch()

if __name__ == '__main__':
    main()
