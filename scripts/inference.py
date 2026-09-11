import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import keras

#config
MODEL_PATH = "brain_tumor_ct_model.keras"
CLASS_NAMES_PATH = "class_names.json"

IMG_SIZE = (224, 224)

#loading model + class names
print("Loading model...")

model = keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("Model loaded successfully.")
print("Classes:", class_names)

#prediction for 1 image
def predict_image(image_path):

    # loading image
    image = Image.open(image_path).convert("RGB")

    # resizing
    image = image.resize(IMG_SIZE)

    # converting image to NumPy array
    image_array = np.array(image, dtype=np.float32)

    # The saved model already rescales raw 0-255 image pixels to MobileNetV2's
    # expected -1 to 1 range. Keep inference preprocessing identical to training.
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    tumor_probability = float(
        model.predict(image_array, verbose=0)[0][0]
    )

    # Our class:
    # 0 = Healthy
    # 1 = Tumor
    if tumor_probability >= 0.5:
        prediction = "Tumor"
        confidence = tumor_probability
    else:
        prediction = "Healthy"
        confidence = 1 - tumor_probability

    return prediction, confidence, tumor_probability

#MAIN
if len(sys.argv) != 2:

    print("\nUsage:")
    print('python scripts/inference.py "path/to/image.jpg"')
    sys.exit(1)


image_path = Path(sys.argv[1])

if not image_path.exists():

    print(f"\nError: Image not found:")
    print(image_path)
    sys.exit(1)


prediction, confidence, tumor_probability = predict_image(
    image_path
)

# RESULT
print("\nBRAIN CT PREDICTION")

print(f"Image      : {image_path}")
print(f"Prediction : {prediction}")
print(f"Confidence : {confidence * 100:.2f}%")

print(f"Tumor score: {tumor_probability * 100:.2f}%")
