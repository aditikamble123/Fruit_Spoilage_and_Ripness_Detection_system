import tensorflow as tf
import numpy as np
from PIL import Image

# Load labels
with open("labels.txt","r",encoding="utf-8") as f:
    labels = [l.strip() for l in f if l.strip()]

# Load TFLite interpreter (from tensorflow package)
interpreter = tf.lite.Interpreter(model_path="fruit_model.tflite")
interpreter.allocate_tensors()

# Get I/O info
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Helper: preprocess one image
def preprocess(path):
    img = Image.open(path).convert("RGB").resize((192,192))  # use your training img_size
    x = np.array(img, dtype=np.float32)/255.0               # match training normalization
    x = np.expand_dims(x, 0)
    return x

# Inference
x = preprocess("test.jpg")  # put a test image in this folder
interpreter.set_tensor(input_details[0]["index"], x)
interpreter.invoke()
y = interpreter.get_tensor(output_details[0]["index"])[0]

cls = int(np.argmax(y))
conf = float(y[cls])
print(f"Prediction: {labels[cls]}  ({conf*100:.1f}% confidence)")
