import json
from pathlib import Path

import tensorflow as tf
import keras
from keras import layers

# config data path and vars
DATASET_DIR = Path("Datasets/Brain Tumor CT scan Images")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 12

MODEL_PATH = "brain_tumor_ct_model.keras"
CLASS_NAMES_PATH = "class_names.json"

#loading dataset
print("Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    # Must match the training loader's shuffle and seed so both subsets are
    # drawn from the same deterministic, class-mixed file order.
    shuffle=True,
)

#naming the classes

class_names = train_ds.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print(f"\nTotal classes: {len(class_names)}")

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=4)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

#data aug
augmentation = keras.Sequential([
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
], name="augmentation")

# MOBILE NET V2
print("\nLoading MobileNetV2...")

base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

# freeze pretrained model
base_model.trainable = False

# Model
inputs = keras.Input(
    shape=(224, 224, 3),
    name="image"
)

x = augmentation(inputs)

# rescaling [0,255] -> [-1,1]
x = layers.Rescaling(
    1 / 127.5,
    offset=-1,
)(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="tumor_probability"
)(x)

model = keras.Model(
    inputs,
    outputs,
    name="BrainTumorMobileNetV2"
)

# 1
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Recall(name="tumor_recall"),
        keras.metrics.Precision(name="tumor_precision"),
        keras.metrics.AUC(name="auc"),
    ],
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=3,
    restore_best_weights=True,
)


print("\nSTAGE 1 — TRAINING CLASSIFIER")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=[early_stopping],
)

print("\nSTAGE 2 — FINE TUNING")

base_model.trainable = True

#early MobileNet layers frozen.
for layer in base_model.layers[:100]:
    layer.trainable = False

#later layers become trainable.
for layer in base_model.layers[100:]:
    layer.trainable = True

# recompiling after changing trainable layers.

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Recall(name="tumor_recall"),
        keras.metrics.Precision(name="tumor_precision"),
        keras.metrics.AUC(name="auc"),
    ],
)

fine_tune_early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=4,
    restore_best_weights=True,
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[fine_tune_early_stopping],
)

#saving
model.save(MODEL_PATH)


print("\nMODEL SAVED\n")

print(f"Model: {MODEL_PATH}")
print(f"Classes: {CLASS_NAMES_PATH}")
