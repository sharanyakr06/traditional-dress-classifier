import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input


# ==========================================
# SETTINGS
# ==========================================

IMAGE_SIZE = (299, 299)

CLASS_NAMES = [
    "Kerala",
    "Punjab",
    "Rajasthan"
]


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading InceptionV3 model...")

model = tf.keras.models.load_model("inceptionv3_best.keras")

print("Model loaded successfully!")


# ==========================================
# GET IMAGE PATH
# ==========================================

image_path = input("\nEnter image path: ")


# ==========================================
# LOAD AND PREPARE IMAGE
# ==========================================

try:

    image = load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image = img_to_array(image)

    # Same preprocessing used during training
    image = preprocess_input(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)


except Exception as e:

    print(f"Error loading image: {e}")
    exit()


# ==========================================
# MAKE PREDICTION
# ==========================================

print("\nMaking prediction...")

predictions = model.predict(image)

predicted_label = np.argmax(predictions[0])

predicted_state = CLASS_NAMES[predicted_label]

confidence = predictions[0][predicted_label] * 100


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n===================================")
print("INCEPTIONV3 PREDICTION")
print("===================================")

print(f"Predicted State: {predicted_state}")
print(f"Confidence: {confidence:.2f}%")


# ==========================================
# CLASS PROBABILITIES
# ==========================================

print("\nClass Probabilities:")

for i, class_name in enumerate(CLASS_NAMES):

    probability = predictions[0][i] * 100

    print(f"{class_name}: {probability:.2f}%")


print("===================================")