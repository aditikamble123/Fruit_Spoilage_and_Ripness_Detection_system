import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model(r"C:\Users\ADITI\projects\Fruit_Ripeness\models\best_fruit_model.h5")

# Export SavedModel directory (for TensorFlow Serving, conversion, etc.)
model.export('models/fruit_savedmodel')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('models/fruit_savedmodel')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('models/fruit_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ SavedModel exported to: models/fruit_savedmodel")
print("✅ TFLite model saved to: models/fruit_model.tflite")
