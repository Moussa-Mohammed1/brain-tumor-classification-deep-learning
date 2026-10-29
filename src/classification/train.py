
import json
from pathlib import Path

import tensorflow_hub as hub
import yaml
import tf_keras as keras

from src.classification.dataset import load_dataset
from src.classification.model import build_model


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "configs" / "classification.yaml"
MODEL_DIR = ROOT_DIR / "models" / "classification"
REPORT_DIR = ROOT_DIR / "reports" / "classification"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    training = config["training"]
    model_config = config["model"]

    keras.utils.set_random_seed(training.get("seed", 42))

 
    train_ds = load_dataset("Train", shuffle=True)
    val_ds = load_dataset("Validation", shuffle=False)
 
    model = build_model(
        num_classes=model_config["num_classes"],
        learning_rate=training["learning_rate"],
        dropout_rate=model_config["dropout_rate"],
    )

    model.summary()

    best_model_path = MODEL_DIR / "best_model.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(REPORT_DIR / "training_log.csv")
        ),
    ]

    # Train and monitor validation performance
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=training["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    # Save the training history for later analysis
    history_path = REPORT_DIR / "training_history.json"

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
            file,
            indent=4,
        )

    print(f"\nBest model saved to: {best_model_path}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
