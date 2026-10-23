from pathlib import Path

import tensorflow as tf 
from tensorflow import keras

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

DATASET_DIR = Path("data/classification_split")

def load_dataset(split_name: str, shuffle: bool = False):

    split_dir = DATASET_DIR / split_name

    if not split_dir.is_dir():
        raise FileNotFoundError(    
            f"dataset not found: {split_dir.resolve()}"
        )

    dataset = keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=IMAGE_SIZE,
        shuffle=shuffle,
        seed=SEED,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        data_format=None,
        format="tf",
        verbose=True,
    )
    normalization = keras.layers.Rescaling(1.0 / 255)
    dataset = dataset.map(
        lambda images, labels:(
            normalization(images),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)

if __name__ == "__main__":
    train_ds = load_dataset("Train", shuffle=True)
    val_ds = load_dataset("Validation")
    test_ds = load_dataset("Test")

    images, labels = next(iter(train_ds))

    print("Image batch shape :", images.shape)
    print("Labels shape      :", labels.shape)
    print("Pixel minimum     :", tf.reduce_min(images).numpy())
    print("Pixel maximum     :", tf.reduce_max(images).numpy())
    print("Classes           :", train_ds.class_names if hasattr(train_ds, "class_names") else "Voir le dossier source")