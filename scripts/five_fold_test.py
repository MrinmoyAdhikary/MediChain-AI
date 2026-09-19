

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import layers


DATASET_DIR = Path("Datasets/Brain Tumor CT scan Images")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
FOLDS = 5
SEED = 42
RESULTS_PATH = Path("five_fold_results.json")
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description="Five-fold CT classifier validation")
    parser.add_argument("--initial-epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=12)
    return parser.parse_args()


def collect_samples(dataset_dir):
    """Return image paths and numeric labels, balanced across folds by class."""
    class_names = sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())
    if len(class_names) != 2:
        raise ValueError(f"Expected exactly two class folders, found: {class_names}")

    paths, labels = [], []
    for label, class_name in enumerate(class_names):
        class_paths = sorted(
            path for path in (dataset_dir / class_name).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(class_paths) < FOLDS:
            raise ValueError(f"{class_name} has fewer than {FOLDS} images.")
        paths.extend(str(path) for path in class_paths)
        labels.extend([label] * len(class_paths))
    return np.array(paths), np.array(labels, dtype=np.float32), class_names


def stratified_fold_ids(labels):
    """Assign folds independently within each class, preserving class ratios."""
    rng = np.random.default_rng(SEED)
    fold_ids = np.empty(len(labels), dtype=np.int32)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        fold_ids[indices] = np.arange(len(indices)) % FOLDS
    return fold_ids


def make_dataset(paths, labels, training):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    def load_image(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMG_SIZE)
        return image, label

    return dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(
        BATCH_SIZE
    ).prefetch(tf.data.AUTOTUNE)


def metrics():
    return [
        "accuracy",
        keras.metrics.Recall(name="tumor_recall"),
        keras.metrics.Precision(name="tumor_precision"),
        keras.metrics.AUC(name="auc"),
    ]


def build_model():
    augmentation = keras.Sequential(
        [layers.RandomRotation(0.03), layers.RandomZoom(0.10), layers.RandomContrast(0.10)],
        name="augmentation",
    )
    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="image")
    x = augmentation(inputs)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="tumor_probability")(x)
    model = keras.Model(inputs, outputs, name="BrainTumorMobileNetV2")
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),  
        loss="binary_crossentropy",
        metrics=metrics(),
    )


def main():
    args = parse_args()
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

    keras.utils.set_random_seed(SEED)
    paths, labels, class_names = collect_samples(DATASET_DIR)
    fold_ids = stratified_fold_ids(labels)
    print(f"Classes: {class_names}; images: {len(paths)}")

    fold_results = []
    for fold in range(FOLDS):
        print(f"\n{'=' * 14} Fold {fold + 1}/{FOLDS} {'=' * 14}")
        train_mask = fold_ids != fold
        validation_mask = ~train_mask
        train_ds = make_dataset(paths[train_mask], labels[train_mask], training=True)
        validation_ds = make_dataset(paths[validation_mask], labels[validation_mask], training=False)

        tf.keras.backend.clear_session()
        keras.utils.set_random_seed(SEED + fold)
        model, base_model = build_model()
        compile_model(model, learning_rate=1e-3)
        model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=args.initial_epochs,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=3, restore_best_weights=True
            )],
            verbose=1,
        )

        base_model.trainable = True
        for layer in base_model.layers[:100]:
            layer.trainable = False
        compile_model(model, learning_rate=1e-5)
        model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=args.fine_tune_epochs,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=4, restore_best_weights=True
            )],
            # Live batch progress, metrics, and an ETA for each epoch.
            verbose=1,
        )

        values = model.evaluate(validation_ds, return_dict=True, verbose=0)
        result = {name: float(value) for name, value in values.items()}
        result["fold"] = fold + 1
        result["train_samples"] = int(train_mask.sum())
        result["validation_samples"] = int(validation_mask.sum())
        fold_results.append(result)
        print("Validation:", ", ".join(f"{key}={value:.4f}" for key, value in values.items()))

    summary = {
        name: float(np.mean([fold[name] for fold in fold_results]))
        for name in ("loss", "accuracy", "tumor_recall", "tumor_precision", "auc")
    }
    output = {"class_names": class_names, "folds": fold_results, "mean": summary}
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nMean validation metrics:")
    for name, value in summary.items():
        print(f"{name}: {value:.4f}")
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
