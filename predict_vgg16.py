import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input


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

print("Loading VGG16 model...")

model = load_model("vgg16_best.keras")

print("Model loaded successfully!")


# ==========================================
# GET IMAGE PATH
# ==========================================

image_path = input("\nEnter image path: ")


# ==========================================
# LOAD AND PREPARE IMAGE
# ==========================================

print("\nLoading image...")

try:

    image = Image.open(image_path).convert("RGB")

    print("Image:", image_path)

    image = image.resize(IMAGE_SIZE)

    image = np.array(image, dtype=np.float32)

    image = preprocess_input(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

except Exception as e:

    print("Error loading image:", e)
    exit()


# ==========================================
# MAKE PREDICTION
# ==========================================

print("Making prediction...")

prediction = model.predict(image, verbose=0)

predicted_index = np.argmax(prediction[0])

predicted_state = CLASS_NAMES[predicted_index]

confidence = prediction[0][predicted_index] * 100


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n===================================")
print("PREDICTION RESULT")
print("===================================")

print(f"Predicted State: {predicted_state}")
print(f"Confidence: {confidence:.2f}%")

print("\nProbabilities:")

for i in range(len(CLASS_NAMES)):

    probability = prediction[0][i] * 100

    print(f"{CLASS_NAMES[i]}: {probability:.2f}%")

print("===================================")