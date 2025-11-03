
import io
from PIL import Image
import numpy as np
import tensorflow as tf

CLASS_NAMES = ["real", "fake"]  # adapt to your classes

_model = None

def load_model(path="new_model.h5"):
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(path)
    return _model

def preprocess(image: Image.Image, img_size=(224,224)):
    image = image.convert("RGB")
    image = image.resize(img_size)
    arr = np.array(image)/255.0
    arr = np.expand_dims(arr, 0)
    return arr

def predict_from_bytes(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes))
    input_arr = preprocess(image)
    model = load_model()
    probs = model.predict(input_arr)[0]
    # If binary sigmoid, probs may be single value
    if probs.shape == ():
        score = float(probs)
        label = CLASS_NAMES[int(score > 0.5)]
        conf = score if label=="fake" else 1-score
    else:
        idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
        conf = float(probs[idx])
    return {"label": label, "confidence": float(conf)}
