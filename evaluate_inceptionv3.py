import pandas as pd
import numpy as np
import os

from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ==========================================
# SETTINGS
# ==========================================

IMAGE_SIZE = (299, 299)

CLASS_NAMES = [
    "Kerala",
    "Punjab",
    "Rajasthan"
]

CLASS_TO_LABEL = {
    "Kerala": 0,
    "Punjab": 1,
    "Rajasthan": 2
}


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading InceptionV3 model...")

model = load_model("inceptionv3_best.keras")

print("Model loaded successfully!")


# ==========================================
# LOAD TEST DATA
# ==========================================

print("\nLoading test data...")

df = pd.read_csv("test.csv")

print("Test samples:", len(df))


# ==========================================
# PREPARE TEST IMAGES
# ==========================================

images = []
true_labels = []

for _, row in df.iterrows():

    state = row["state"]
    image_name = row["image_name"]

    folder = state.lower()

    image_path = os.path.join(folder, image_name)

    try:

        image = Image.open(image_path).convert("RGB")

        image = image.resize(IMAGE_SIZE)

        image = np.array(image, dtype=np.float32)

        # Same preprocessing used for InceptionV3
        image = preprocess_input(image)

        images.append(image)

        true_labels.append(CLASS_TO_LABEL[state])

    except Exception as e:

        print(f"Error loading {image_path}: {e}")


X_test = np.array(images)
y_test = np.array(true_labels)


print("Images prepared:", X_test.shape)


# ==========================================
# MAKE PREDICTIONS
# ==========================================

print("\nMaking predictions...")

predictions = model.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

print(
    classification_report(
        y_test,
        predicted_labels,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

cm = confusion_matrix(
    y_test,
    predicted_labels
)

print(cm)


# ==========================================
# TEST ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    predicted_labels
)

print("\n===================================")
print(f"TEST ACCURACY: {accuracy * 100:.2f}%")
print("===================================")