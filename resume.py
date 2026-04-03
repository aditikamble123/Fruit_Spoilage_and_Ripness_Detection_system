# resume.py — continue fine-tuning from a saved .keras checkpoint

import os
import tensorflow as tf

# ---------- Paths and config (adjust if needed) ----------
dataset_root = r"C:\Users\ADITI\projects\Fruit_Ripeness"  # parent of Train/ and optional Val/
train_dir = os.path.join(dataset_root, "Train")
val_dir   = os.path.join(dataset_root, "Val")  # set to None if not created
img_size  = (224, 224)
batch_size = 32
seed = 1337

ckpt_path = r"models\fruit_fresh_spoiled_best.keras"  # produced by ModelCheckpoint
label_smoothing = 0.05
resume_epochs = 6  # how many more epochs to run

# ---------- Build datasets (same split/seed as training) ----------
if val_dir and os.path.isdir(val_dir):
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size,
        labels="inferred", label_mode="int", shuffle=True, seed=seed
    )
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=img_size, batch_size=batch_size,
        labels="inferred", label_mode="int", shuffle=True, seed=seed
    )
else:
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size,
        labels="inferred", label_mode="int",
        validation_split=0.2, subset="training", seed=seed, shuffle=True
    )
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size,
        labels="inferred", label_mode="int",
        validation_split=0.2, subset="validation", seed=seed, shuffle=True
    )

# capture class_names BEFORE prefetch/cache
class_names = train_ds_raw.class_names
num_classes = len(class_names)

# performance pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds_raw.cache().prefetch(AUTOTUNE)
val_ds   = val_ds_raw.cache().prefetch(AUTOTUNE)

# if earlier training used one-hot with label smoothing, keep it consistent
train_ds_oh = train_ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))
val_ds_oh   = val_ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))

# ---------- Load checkpoint and compile ----------
# load the best/last checkpoint (.keras format includes optimizer state in Keras 3)
model = tf.keras.models.load_model(ckpt_path)

# IMPORTANT: compile with a plain float LR to avoid the "LearningRateSchedule not settable" crash
opt = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
    metrics=["accuracy"]
)

# ---------- Resume training ----------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max"),
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
]

model.fit(
    train_ds_oh,
    validation_data=val_ds_oh,
    epochs=resume_epochs,
    callbacks=callbacks
)

# Optional: export SavedModel/TFLite again after resuming
# savedmodel = "models/fruit_savedmodel"
# model.save(savedmodel)
# converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# open("models/fruit_model.tflite", "wb").write(converter.convert())
