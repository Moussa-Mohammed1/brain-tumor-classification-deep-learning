import json
import shutil
from pathlib import Path

import yaml
import tf_keras as keras

from src.classification.dataset import load_dataset
from src.classification.model import build_model, set_backbone_trainable


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "configs" / "classification.yaml"
MODEL_DIR = ROOT_DIR / "models" / "classification"
REPORT_DIR = ROOT_DIR / "reports" / "classification"


def make_callbacks(
    checkpoint_path: Path,
    log_path: Path,
    early_stopping_patience: int,
    reduce_lr_patience: int,
):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(log_path)),
    ]


def save_history(history1, history2, path: Path) -> None:
    """Merge the two training phases into a single history JSON."""
    merged = {
        key: [float(value) for value in values]
        for key, values in history1.history.items()
    }

    if history2 is not None:
        for key, values in history2.history.items():
            merged.setdefault(key, []).extend(
                float(value) for value in values
            )

    with open(path, "w", encoding="utf-8") as file:
        json.dump(merged, file, indent=4)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_config = config["model"]
    training = config["training"]
    finetuning = config.get("finetuning", {})
    finetune_enabled = bool(finetuning.get("enabled", False))

    keras.utils.set_random_seed(training.get("seed", 42))

    train_ds = load_dataset("Train", shuffle=True)
    val_ds = load_dataset("Validation", shuffle=False)

    # ------------------------------------------------------------------
    # Phase 1 - feature extraction: the InceptionV3 backbone stays frozen
    # and only the classification head is trained.
    # ------------------------------------------------------------------
    model = build_model(
        num_classes=model_config["num_classes"],
        learning_rate=training["learning_rate"],
        dropout_rate=model_config["dropout_rate"],
        trainable=False,
    )
    model.summary()

    phase1_path = MODEL_DIR / "best_model_phase1.keras"
    phase1_log = REPORT_DIR / "training_log_phase1.csv"

    print("\n=== Phase 1: feature extraction (frozen backbone) ===")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=training["epochs"],
        callbacks=make_callbacks(
            checkpoint_path=phase1_path,
            log_path=phase1_log,
            early_stopping_patience=finetuning.get(
                "phase1_patience", 5
            ),
            reduce_lr_patience=finetuning.get(
                "phase1_reduce_lr_patience", 2
            ),
        ),
        verbose=1,
    )

    best_val_loss = min(history1.history["val_loss"])
    best_phase = 1
    best_path = phase1_path
    history2 = None

    # ------------------------------------------------------------------
    # Phase 2 - fine-tuning: unfreeze the backbone and continue training
    # with a much smaller learning rate. The better of the two phases is
    # exported as best_model.keras.
    # ------------------------------------------------------------------
    if finetune_enabled:
        print("\n=== Phase 2: fine-tuning (trainable backbone) ===")
        set_backbone_trainable(model, True)

        finetune_learning_rate = finetuning.get("learning_rate", 1e-5)
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=finetune_learning_rate
            ),
            loss=training.get(
                "loss", "sparse_categorical_crossentropy"
            ),
            metrics=training.get("metrics", ["accuracy"]),
        )

        phase2_path = MODEL_DIR / "best_model_phase2.keras"
        phase2_log = REPORT_DIR / "training_log_phase2.csv"

        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=finetuning.get("epochs", 10),
            callbacks=make_callbacks(
                checkpoint_path=phase2_path,
                log_path=phase2_log,
                early_stopping_patience=finetuning.get(
                    "early_stopping_patience", 3
                ),
                reduce_lr_patience=finetuning.get(
                    "reduce_lr_patience", 2
                ),
            ),
            verbose=1,
        )

        phase2_val_loss = min(history2.history["val_loss"])

        print(f"\nPhase 1 best val_loss: {best_val_loss:.4f}")
        print(f"Phase 2 best val_loss: {phase2_val_loss:.4f}")

        if phase2_val_loss < best_val_loss:
            best_val_loss = phase2_val_loss
            best_phase = 2
            best_path = phase2_path

    # evaluate.py always reads models/classification/best_model.keras
    final_path = MODEL_DIR / "best_model.keras"
    shutil.copy2(best_path, final_path)

    history_path = REPORT_DIR / "training_history.json"
    save_history(history1, history2, history_path)

    print(
        f"\nBest model (phase {best_phase}, "
        f"val_loss={best_val_loss:.4f}) saved to: {final_path}"
    )
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
