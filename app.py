from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge


IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Kerala", "Punjab", "Rajasthan"]
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "resnet50_augmented_finetuned_best.keras"
)
MAX_UPLOAD_SIZE = 4 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

app = Flask(__name__, static_folder="public", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
classifier = None


def load_classifier():
    global classifier
    if classifier is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"The model file is missing: {MODEL_PATH.name}. "
                "Add it to the project folder before starting the app."
            )
        classifier = tf.keras.models.load_model(MODEL_PATH)
    return classifier


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():
    uploaded_file = request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify(error="Choose an image before analyzing it."), 400

    if Path(uploaded_file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        return jsonify(error="Please upload a JPG, JPEG, or PNG image."), 400

    try:
        image = Image.open(uploaded_file.stream).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="This file could not be opened as an image."), 400

    resized = image.resize(IMAGE_SIZE)
    model_input = np.asarray(resized, dtype=np.float32)
    model_input = preprocess_input(model_input)
    model_input = np.expand_dims(model_input, axis=0)
    scores = load_classifier().predict(model_input, verbose=0)[0]
    predicted_index = int(np.argmax(scores))

    return jsonify(prediction=CLASS_NAMES[predicted_index])


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    return jsonify(error="The image is too large. Maximum upload size is 4 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
