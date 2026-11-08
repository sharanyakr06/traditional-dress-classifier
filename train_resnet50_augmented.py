import pandas as pd
import numpy as np
import os

from PIL import Image

import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

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

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ==========================================
# SETTINGS
# ==========================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 16

EPOCHS_HEAD = 15
EPOCHS_FINE = 15

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

print(
    "Training:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Testing:",
    X_test.shape
)


# ==========================================
# DATA AUGMENTATION
# ==========================================

print("\nCreating data augmentation pipeline...")


train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=15,

    width_shift_range=0.08,

    height_shift_range=0.08,

    zoom_range=0.10,

    shear_range=0.08,

    horizontal_flip=True,

    fill_mode="nearest"
)


# Validation and testing:
# NO augmentation

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


train_generator = train_datagen.flow(

    X_train,
    y_train,

    batch_size=BATCH_SIZE,

    shuffle=True
)


val_generator = val_datagen.flow(

    X_val,
    y_val,

    batch_size=BATCH_SIZE,

    shuffle=False
)


test_generator = test_datagen.flow(

    X_test,
    y_test,

    batch_size=BATCH_SIZE,

    shuffle=False
)


# ==========================================
# CREATE RESNET50
# ==========================================

print("\nCreating augmented ResNet50 model...")


base_model = ResNet50(

    weights="imagenet",

    include_top=False,

    input_shape=(224, 224, 3)
)


# Initially freeze ResNet50

base_model.trainable = False


# ==========================================
# CLASSIFICATION HEAD
# ==========================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(
    0.5
)(x)

output = Dense(
    3,
    activation="softmax"
)(x)


model = Model(

    inputs=base_model.input,

    outputs=output
)


# ==========================================
# STAGE 1 COMPILE
# ==========================================

model.compile(

    optimizer=Adam(
        learning_rate=1e-4
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ==========================================
# STAGE 1 CALLBACKS
# ==========================================

checkpoint = ModelCheckpoint(

    "resnet50_augmented_stage1_best.keras",

    monitor="val_loss",

    save_best_only=True,

    mode="min",

    verbose=1
)


early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=4,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=1e-7,

    verbose=1
)


# ==========================================
# STAGE 1
# TRAIN CLASSIFICATION HEAD
# ==========================================

print("\n===================================")
print("STAGE 1: AUGMENTED TRAINING")
print("===================================")


history_head = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=EPOCHS_HEAD,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


print("\nStage 1 completed!")

print(
    "Best Stage 1 validation accuracy:",
    max(history_head.history["val_accuracy"]) * 100
)


# ==========================================
# IMPORTANT:
# DO NOT LOAD THE .KERAS MODEL HERE
#
# We continue using the same model and
# same base_model so that the ResNet50
# layers remain accessible.
# ==========================================


# ==========================================
# STAGE 2
# FINE-TUNING
# ==========================================

print("\n===================================")
print("STAGE 2: FINE-TUNING RESNET50")
print("===================================")


# Unfreeze the ResNet50

base_model.trainable = True


# Freeze all ResNet50 layers first

for layer in base_model.layers:

    layer.trainable = False


# Unfreeze only the last 30 layers

for layer in base_model.layers[-30:]:

    # Keep BatchNormalization frozen
    # because dataset is small

    if not isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = True


print("\nFine-tuning layers:")

trainable_count = 0
frozen_count = 0

for layer in base_model.layers:

    if layer.trainable:

        trainable_count += 1

    else:

        frozen_count += 1


print(
    "Trainable ResNet50 layers:",
    trainable_count
)

print(
    "Frozen ResNet50 layers:",
    frozen_count
)


# ==========================================
# RECOMPILE FOR FINE-TUNING
# ==========================================

model.compile(

    optimizer=Adam(

        learning_rate=1e-5
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ==========================================
# FINE-TUNING CALLBACKS
# ==========================================

fine_tune_checkpoint = ModelCheckpoint(

    "resnet50_augmented_finetuned_best.keras",

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1
)


fine_tune_early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


fine_tune_reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=1e-7,

    verbose=1
)


# ==========================================
# STAGE 2 TRAINING
# ==========================================

history_fine = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=EPOCHS_FINE,

    callbacks=[

        fine_tune_checkpoint,

        fine_tune_early_stopping,

        fine_tune_reduce_lr

    ]
)


# ==========================================
# FINAL TEST EVALUATION
# ==========================================

print("\n===================================")
print("BEST CHECKPOINT TEST EVALUATION")
print("===================================")

best_model = tf.keras.models.load_model(
    "resnet50_augmented_finetuned_best.keras"
)

test_loss, test_accuracy = best_model.evaluate(

    test_generator,

    verbose=1
)


print("\n===================================")
print("AUGMENTED RESNET50 RESULTS")
print("===================================")

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# ==========================================
# SAVE FINAL MODEL
# ==========================================

best_model.save(

    "resnet50_augmented_finetuned_final.keras"
)


print("\nModel saved successfully!")

print(
    "resnet50_augmented_stage1_best.keras"
)

print(
    "resnet50_augmented_finetuned_best.keras"
)

print(
    "resnet50_augmented_finetuned_final.keras"
)
