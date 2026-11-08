import pandas as pd
import numpy as np
import os

from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ==========================================
# SETTINGS
# ==========================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

CLASS_NAMES = ["Kerala", "Punjab", "Rajasthan"]

CLASS_TO_LABEL = {
    "Kerala": 0,
    "Punjab": 1,
    "Rajasthan": 2
}


# ==========================================
# LOAD IMAGES
# ==========================================

def load_images(csv_file):

    df = pd.read_csv(csv_file)

    images = []
    labels = []

    for _, row in df.iterrows():

        state = row["state"]
        image_name = row["image_name"]

        folder = state.lower()

        image_path = os.path.join(folder, image_name)

        try:

            image = Image.open(image_path).convert("RGB")

            image = image.resize(IMAGE_SIZE)

            image = np.array(image, dtype=np.float32)

            image = preprocess_input(image)

            images.append(image)

            labels.append(CLASS_TO_LABEL[state])

        except Exception as e:

            print(f"Error loading {image_path}: {e}")

    return np.array(images), np.array(labels)


# ==========================================
# LOAD TRAINING DATA
# ==========================================

print("Loading training data...")

X_train, y_train = load_images("train.csv")

print("Loading validation data...")

X_val, y_val = load_images("validation.csv")

print("Loading testing data...")

X_test, y_test = load_images("test.csv")


print("\nData loaded successfully!")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ==========================================
# CREATE RESNET50 MODEL
# ==========================================

print("\nCreating ResNet50 model...")


# Load ResNet50 with ImageNet knowledge
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# Freeze ResNet50 layers initially
base_model.trainable = False


# Add our own classification layers
x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(128, activation="relu")(x)

x = Dropout(0.5)(x)

output = Dense(
    3,
    activation="softmax"
)(x)


model = Model(
    inputs=base_model.input,
    outputs=output
)


# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# SHOW MODEL
# ==========================================

model.summary()


# ==========================================
# CALLBACKS
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


checkpoint = ModelCheckpoint(
    "resnet50_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)


# ==========================================
# TRAIN MODEL
# ==========================================

print("\nStarting ResNet50 training...\n")


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        early_stopping,
        checkpoint
    ],
    shuffle=True
)


# ==========================================
# TEST MODEL
# ==========================================

print("\nEvaluating model on test data...")


test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)


print("\n===================================")
print("RESNET50 RESULTS")
print("===================================")

print(f"Test Loss: {test_loss:.4f}")

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ==========================================
# SAVE FINAL MODEL
# ==========================================

model.save("resnet50_final.keras")

print("\nModel saved successfully!")
print("resnet50_final.keras")
print("resnet50_best.keras")