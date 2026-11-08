import numpy as np
import tensorflow as tf

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


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading MobileNetV2 model...")

model = tf.keras.models.load_model("mobilenetv2_best.keras")

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

    print(f"\nError loading image: {e}")
    exit()


# ==========================================
# MAKE PREDICTION
# ==========================================

print("\nMaking prediction...")

prediction = model.predict(image, verbose=0)

predicted_label = np.argmax(prediction[0])

predicted_class = CLASS_NAMES[predicted_label]

confidence = prediction[0][predicted_label] * 100


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n===================================")
print("MOBILENETV2 PREDICTION")
print("===================================")

print(f"Predicted State: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")


# ==========================================
# DISPLAY ALL PROBABILITIES
# ==========================================

print("\nClass Probabilities:")

for i, class_name in enumerate(CLASS_NAMES):

    probability = prediction[0][i] * 100

    print(f"{class_name}: {probability:.2f}%")

print("===================================")