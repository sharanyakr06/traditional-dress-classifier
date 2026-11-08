import csv
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix


IMAGE_SIZE = (224, 224)
MODEL_PATH = "resnet50_improved_best.keras"
TEST_CSV = "test.csv"
RESULTS_JSON = "resnet50_improved_results.json"
CONFUSION_CSV = "resnet50_improved_confusion_matrix.csv"
CLASS_NAMES = ["Kerala", "Punjab", "Rajasthan"]
CLASS_TO_LABEL = {name: index for index, name in enumerate(CLASS_NAMES)}


def load_test_images():
    images = []
    labels = []
    test_df = pd.read_csv(TEST_CSV)

    for _, row in test_df.iterrows():
        state = row["state"]
        image_path = os.path.join(state.lower(), row["image_name"])
        image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
        images.append(np.asarray(image, dtype=np.float32))
        labels.append(CLASS_TO_LABEL[state])

    return np.asarray(images), np.asarray(labels)


print(f"Loading {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
X_test, y_true = load_test_images()
predictions = model.predict(X_test, verbose=1)
y_pred = np.argmax(predictions, axis=1)

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)
matrix = confusion_matrix(y_true, y_pred)

results = {
    "model": MODEL_PATH,
    "test_csv": TEST_CSV,
    "test_samples": int(len(y_true)),
    "accuracy": float(report["accuracy"]),
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
}

with open(RESULTS_JSON, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

with open(CONFUSION_CSV, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["actual/predicted", *CLASS_NAMES])
    for class_name, row in zip(CLASS_NAMES, matrix):
        writer.writerow([class_name, *row.tolist()])

print("\nImproved ResNet50 results")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
print("Confusion matrix:")
print(matrix)
print(f"Saved: {RESULTS_JSON}")
print(f"Saved: {CONFUSION_CSV}")