import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fast pose event model.")
    parser.add_argument("--dataset", required=True, help="Path to NPZ produced by prepare_poselift_windows.py")
    parser.add_argument("--model-out", default="models/pose_event_detector.keras", help="Output model path")
    parser.add_argument("--labels-out", default="models/pose_event_labels.json", help="Output labels json path")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def build_model(window: int, features: int, num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window, features))
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    args = parse_args()
    data = np.load(args.dataset, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    labels = [str(x) for x in data["labels"].tolist()]

    if len(X) == 0:
        raise ValueError("Empty dataset after preprocessing.")

    split = int(0.85 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    if len(X_val) == 0:
        X_val, y_val = X_train, y_train

    model = build_model(window=X.shape[1], features=X.shape[2], num_classes=len(labels))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ]
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    model_out = Path(args.model_out)
    labels_out = Path(args.labels_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_out)
    labels_out.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"Saved model: {model_out}")
    print(f"Saved labels: {labels_out}")


if __name__ == "__main__":
    main()
