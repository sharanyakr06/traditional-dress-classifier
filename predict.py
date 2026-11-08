import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

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

print("Loading fine-tuned augmented ResNet50 model...")

model = tf.keras.models.load_model("resnet50_augmented_finetuned_best.keras")

print("Model loaded successfully!")


# ==========================================
# PREDICT FUNCTION
# ==========================================

def predict_image(image_path):

    print("\nLoading image...")
    print("Image:", image_path)

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Resize to ResNet50 input size
    image = image.resize(IMAGE_SIZE)

    # Convert to NumPy array
    image = np.array(image, dtype=np.float32)
    image = preprocess_input(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    print("Making prediction...")

    # Predict
    predictions = model.predict(image, verbose=0)

    # Get predicted class
    predicted_index = np.argmax(predictions[0])

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = predictions[0][predicted_index] * 100

    print("\n===================================")
    print("PREDICTION RESULT")
    print("===================================")

    print("Predicted State:", predicted_class)
    print(f"Confidence: {confidence:.2f}%")

    print("\nProbabilities:")

    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {predictions[0][i] * 100:.2f}%")

    print("===================================")


# ==========================================
# MAIN PROGRAM
# ==========================================

image_path = input("\nEnter image path: ")

predict_image(image_path)