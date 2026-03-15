import os
import argparse
import json

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint


def build_model(input_shape=(128, 128, 3), num_classes=3):
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main(args):
    train_dir = args.train_dir
    valid_dir = args.valid_dir
    output_path = args.output

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    valid_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=args.batch_size,
        class_mode="categorical",
    )

    valid_data = valid_gen.flow_from_directory(
        valid_dir,
        target_size=(128, 128),
        batch_size=args.batch_size,
        class_mode="categorical",
    )

    model = build_model()

    # Save class indices for consistent inference label mapping
    class_index_path = os.path.join(os.path.dirname(output_path), "class_indices.json")
    os.makedirs(os.path.dirname(class_index_path), exist_ok=True)
    with open(class_index_path, "w", encoding="utf-8") as f:
        json.dump(train_data.class_indices, f, indent=2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    checkpoint = ModelCheckpoint(output_path, monitor="val_accuracy", save_best_only=True, mode="max")

    model.fit(
        train_data,
        epochs=args.epochs,
        validation_data=valid_data,
        callbacks=[checkpoint],
    )

    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="../dataset_std/train")
    parser.add_argument("--valid-dir", default="../dataset_std/valid")
    parser.add_argument("--output", default="./lung_cancer_model.h5")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    main(args)
