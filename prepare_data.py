import pandas as pd
import numpy as np
import os
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Image size required by ResNet50
IMAGE_SIZE = (224, 224)

# State labels
class_names = ["Kerala", "Punjab", "Rajasthan"]

# Convert state name to number
class_to_label = {
    "Kerala": 0,
    "Punjab": 1,
    "Rajasthan": 2
}


def load_images(csv_file):
    df = pd.read_csv(csv_file)

    images = []
    labels = []

    for _, row in df.iterrows():

        state = row["state"]
        image_name = row["image_name"]

        # Folder name is lowercase
        folder = state.lower()

        image_path = os.path.join(folder, image_name)

        try:
            # Open image
            image = Image.open(image_path).convert("RGB")

            # Resize to 224x224
            image = image.resize(IMAGE_SIZE)

            # Convert image to NumPy array
            image = np.array(image, dtype=np.float32)

            # ResNet50 preprocessing
            image = preprocess_input(image)

            images.append(image)
            labels.append(class_to_label[state])

        except Exception as e:
            print(f"Error loading {image_path}: {e}")

    return np.array(images), np.array(labels)


# Load all three datasets
print("Loading training images...")
X_train, y_train = load_images("train.csv")

print("Loading validation images...")
X_val, y_val = load_images("validation.csv")

print("Loading testing images...")
X_test, y_test = load_images("test.csv")


# Display results
print("\n==============================")
print("DATA PREPARATION COMPLETE ✅")
print("==============================")

print("\nTraining images:", len(X_train))
print("Validation images:", len(X_val))
print("Testing images:", len(X_test))

print("\nImage shape:", X_train.shape[1:])
print("Training labels shape:", y_train.shape)

print("\nClasses:")
print(class_names)

print("\nFirst 10 labels:")
print(y_train[:10])