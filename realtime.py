import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

# Load labels
with open("labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="fruit_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded. Labels:", labels)

# Image size used during training
IMG_SIZE = (192, 192)  # adjust to your training size (160/192/224)

# Confidence threshold for rejecting non-fruit
CONF_THRESHOLD = 0.7  # Adjust between 0.6-0.9 for strictness

# Optional temperature scaling for calibration (T > 1 softens overconfident predictions)
TEMP = 1.5

def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    return x

# Softmax with temperature scaling
def softmax_with_temp(logits, T=1.0):
    z = logits / T
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Camera opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    x = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details[0]['index'])[0]
    probs = softmax_with_temp(logits, TEMP)

    cls_idx = int(np.argmax(probs))
    confidence = float(probs[cls_idx])
    label = labels[cls_idx]

    # Reject prediction if confidence is below threshold
    if confidence < CONF_THRESHOLD:
        text = f"Unknown / Not a fruit ({confidence*100:.1f}%)"
        color = (0, 255, 255)  # Yellow for uncertain
    else:
        color = (0, 255, 0) if 'fresh' in label.lower() else (0, 0, 255)
        text = f"{label}: {confidence*100:.1f}%"

    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, cv2.LINE_AA)
    cv2.imshow('Fruit Ripeness Classifier', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
