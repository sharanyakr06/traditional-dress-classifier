import pandas as pd
import numpy as np
import os

from PIL import Image

import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout
)

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ==========================================
# SETTINGS
# ==========================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 16

STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 15

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

        image_path = os.path.join(
            folder,
            image_name
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image = image.resize(
                IMAGE_SIZE
            )

            image = np.array(
                image,
                dtype=np.float32
            )

            images.append(image)

            labels.append(
                CLASS_TO_LABEL[state]
            )

        except Exception as e:

            print(
                f"Error loading {image_path}: {e}"
            )

    return (
        np.array(images),
        np.array(labels)
    )


# ==========================================
# LOAD DATA
# ==========================================

print("Loading training data...")

X_train, y_train = load_images(
    "train.csv"
)

print("Loading validation data...")

X_val, y_val = load_images(
    "validation.csv"
)

print("Loading testing data...")

X_test, y_test = load_images(
    "test.csv"
)


print("\nData loaded successfully!")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ==========================================
# DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.08
    ),

    layers.RandomZoom(
        0.10
    ),

    layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),

    layers.RandomContrast(
        0.10
    )

], name="data_augmentation")


# ==========================================
# CREATE RESNET50
# ==========================================

print("\nCreating improved ResNet50 model...")


base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# ==========================================
# STAGE 1
# FREEZE RESNET50
# ==========================================

base_model.trainable = False


# ==========================================
# MODEL INPUT
# ==========================================

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)


# Apply augmentation only during training

x = data_augmentation(
    inputs
)


# ResNet50 preprocessing

x = preprocess_input(
    x
)


# ResNet50 feature extraction

x = base_model(
    x,
    training=False
)


# ==========================================
# CLASSIFICATION HEAD
# ==========================================

x = GlobalAveragePooling2D()(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(
    0.5
)(x)

outputs = Dense(
    3,
    activation="softmax"
)(x)


model = Model(
    inputs=inputs,
    outputs=outputs
)


# ==========================================
# COMPILE STAGE 1
# ==========================================

model.compile(

    optimizer=Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


print("\n===================================")
print("STAGE 1: TRAINING CLASSIFIER HEAD")
print("===================================")


# ==========================================
# CALLBACKS - STAGE 1
# ==========================================

stage1_checkpoint = ModelCheckpoint(
    "resnet50_improved_stage1.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

stage1_early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)


# ==========================================
# STAGE 1 TRAINING
# ==========================================

history_stage1 = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=STAGE1_EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        stage1_checkpoint,
        stage1_early_stopping
    ],

    shuffle=True
)


# ==========================================
# STAGE 2
# FINE-TUNING
# ==========================================

print("\n===================================")
print("STAGE 2: FINE-TUNING RESNET50")
print("===================================")


# Unfreeze ResNet50

base_model.trainable = True


# Freeze earlier layers
# Fine-tune only the last 30 layers

for layer in base_model.layers[:-30]:

    layer.trainable = False


# Keep BatchNormalization layers frozen

for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False


# ==========================================
# COMPILE FOR FINE-TUNING
# ==========================================

model.compile(

    optimizer=Adam(
        learning_rate=0.00001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ==========================================
# CALLBACKS - STAGE 2
# ==========================================

stage2_checkpoint = ModelCheckpoint(
    "resnet50_improved_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

stage2_early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7
)


# ==========================================
# STAGE 2 TRAINING
# ==========================================

history_stage2 = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=STAGE2_EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        stage2_checkpoint,
        stage2_early_stopping,
        reduce_lr
    ],

    shuffle=True
)


# ==========================================
# LOAD BEST MODEL
# ==========================================

print("\nLoading best fine-tuned model...")

best_model = tf.keras.models.load_model(
    "resnet50_improved_best.keras"
)


print("Best model loaded successfully!")


# ==========================================
# FINAL TEST EVALUATION
# ==========================================

print("\n===================================")
print("FINAL TEST EVALUATION")
print("===================================")


test_loss, test_accuracy = best_model.evaluate(

    X_test,
    y_test,

    verbose=1
)


print("\n===================================")
print("IMPROVED RESNET50 RESULTS")
print("===================================")

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ==========================================
# SAVE FINAL MODEL
# ==========================================

best_model.save(
    "resnet50_improved_final.keras"
)


print("\nModel saved successfully!")

print(
    "resnet50_improved_best.keras"
)

print(
    "resnet50_improved_final.keras"
)