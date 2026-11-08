import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==========================================
# SETTINGS
# ==========================================

IMAGE_SIZE = (224, 224)

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

print("Loading MobileNetV2 model...")

model = tf.keras.models.load_model("mobilenetv2_best.keras")

print("Model loaded successfully!")


# ==========================================
# LOAD TEST DATA
# ==========================================

print("\nLoading test data...")

test_df = pd.read_csv("test.csv")

print("Test samples:", len(test_df))


# ==========================================
# PREPARE TEST IMAGES
# ==========================================

images = []
true_labels = []

for _, row in test_df.iterrows():

    state = row["state"]
    image_name = row["image_name"]

    # Match the exact folder structure
    image_path = state.lower() + "/" + image_name

    try:

        image = load_img(
            image_path,
            target_size=IMAGE_SIZE
        )

        image = img_to_array(image)

        # IMPORTANT:
        # Same preprocessing used during MobileNetV2 training
        image = preprocess_input(image)

        images.append(image)

        true_labels.append(
            CLASS_TO_LABEL[state]
        )

    except Exception as e:

        print(f"Error loading {image_path}: {e}")


X_test = np.array(images)
y_true = np.array(true_labels)


print("Images prepared:", X_test.shape)


# ==========================================
# MAKE PREDICTIONS
# ==========================================

print("\nMaking predictions...")

predictions = model.predict(X_test)

y_pred = np.argmax(
    predictions,
    axis=1
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

print(
    classification_report(
        y_true,
        y_pred,
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
    y_true,
    y_pred
)

print(cm)


# ==========================================
# TEST ACCURACY
# ==========================================

accuracy = np.mean(
    y_true == y_pred
)

print("\n===================================")
print(f"TEST ACCURACY: {accuracy * 100:.2f}%")
print("===================================")