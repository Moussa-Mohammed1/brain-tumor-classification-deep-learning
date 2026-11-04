import json
import shutil
from pathlib import Path

import yaml
import tf_keras as keras

from src.classification.dataset import load_dataset
from src.classification.model import build_model, set_backbone_trainable
from src.tracking.mlflow_utils import MLflowTracker, utc_timestamp


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "configs" / "classification.yaml"
MODEL_DIR = ROOT_DIR / "models" / "classification"
REPORT_DIR = ROOT_DIR / "reports" / "classification"


class MlflowEpochLogger(keras.callbacks.Callback):
    """Log Keras epoch metrics to MLflow under a phase-specific prefix."""

    def __init__(self, tracker: MLflowTracker, prefix: str):
        super().__init__()
        self.tracker = tracker
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            self.tracker.log_metrics(
                logs,
                prefix=self.prefix,
                step=epoch,
            )


def make_callbacks(
    checkpoint_path: Path,
    log_path: Path,
    early_stopping_patience: int,
    reduce_lr_patience: int,
    tracker: MLflowTracker | None = None,
    phase_prefix: str = "",
):
    callbacks = [
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
    if tracker is not None:
        callbacks.append(MlflowEpochLogger(tracker, phase_prefix))
    return callbacks


def count_images(split_name: str) -> int:
    """Count image files in ``data/classification_split/<split_name>``."""
    split_dir = ROOT_DIR / "data" / "classification_split" / split_name
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sum(
        1
        for path in split_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def load_class_names(split_name: str = "Train") -> list:
    """Return the sorted class names (same order as ``load_dataset``)."""
    split_dir = ROOT_DIR / "data" / "classification_split" / split_name
    return sorted(
        path.name for path in split_dir.iterdir() if path.is_dir()
    )


def save_history(history1, history2, path: Path) -> dict:
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

    return merged


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_config = config["model"]
    training = config["training"]
    finetuning = config.get("finetuning", {})
    finetune_enabled = bool(finetuning.get("enabled", False))
    mlflow_config = config.get("mlflow", {})

    keras.utils.set_random_seed(training.get("seed", 42))

    train_ds = load_dataset("Train", shuffle=True)
    val_ds = load_dataset("Validation", shuffle=False)
    class_names = load_class_names("Train")

    with MLflowTracker(
        mlflow_config,
        default_experiment="brainscan-classification",
        run_name=f"classification-{utc_timestamp()}",
        tags={
            "stage": "training",
            "task": "classification",
            "model": model_config.get("name", "inception_v3"),
        },
    ) as tracker:
        tracker.log_params(
            {
                "model": model_config,
                "training": training,
                "finetuning": finetuning,
                "data": {
                    "train": str(
                        ROOT_DIR / "data" / "classification_split" / "Train"
                    ),
                    "validation": str(
                        ROOT_DIR
                        / "data"
                        / "classification_split"
                        / "Validation"
                    ),
                    "classes": class_names,
                    "train_images": count_images("Train"),
                    "validation_images": count_images("Validation"),
                },
            }
        )
        tracker.log_artifacts(CONFIG_PATH, artifact_path="config")

        # --------------------------------------------------------------
        # Phase 1 - feature extraction: the InceptionV3 backbone stays
        # frozen and only the classification head is trained.
        # --------------------------------------------------------------
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
                tracker=tracker,
                phase_prefix="phase1/",
            ),
            verbose=1,
        )

        best_val_loss = min(history1.history["val_loss"])
        best_phase = 1
        best_path = phase1_path
        history2 = None
        phase2_log = REPORT_DIR / "training_log_phase2.csv"

        tracker.log_metrics(
            {
                "best/val_loss": best_val_loss,
                "best/epoch": (
                    history1.history["val_loss"].index(best_val_loss) + 1
                ),
            },
            prefix="phase1/",
        )

        # --------------------------------------------------------------
        # Phase 2 - fine-tuning: unfreeze the backbone and continue
        # training with a much smaller learning rate. The better of the
        # two phases is exported as best_model.keras.
        # --------------------------------------------------------------
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
                    tracker=tracker,
                    phase_prefix="phase2/",
                ),
                verbose=1,
            )

            phase2_val_loss = min(history2.history["val_loss"])

            print(f"\nPhase 1 best val_loss: {best_val_loss:.4f}")
            print(f"Phase 2 best val_loss: {phase2_val_loss:.4f}")

            tracker.log_metrics(
                {
                    "best/val_loss": phase2_val_loss,
                    "best/epoch": (
                        history2.history["val_loss"].index(phase2_val_loss)
                        + 1
                    ),
                },
                prefix="phase2/",
            )

            if phase2_val_loss < best_val_loss:
                best_val_loss = phase2_val_loss
                best_phase = 2
                best_path = phase2_path

        # evaluate.py always reads models/classification/best_model.keras
        final_path = MODEL_DIR / "best_model.keras"
        shutil.copy2(best_path, final_path)

        history_path = REPORT_DIR / "training_history.json"
        merged_history = save_history(history1, history2, history_path)

        tracker.log_metrics(
            {
                "best/val_loss": best_val_loss,
                "best/phase": best_phase,
            }
        )
        tracker.set_tags(
            {
                "best_phase": best_phase,
                "classes": ",".join(class_names),
            }
        )
        tracker.log_artifacts(
            [phase1_log, phase2_log, history_path],
            artifact_path="reports",
        )
        tracker.log_dict(
            {
                "model_name": model_config.get("name"),
                "architecture": "InceptionV3 (TF Hub) + Dense softmax",
                "image_size": model_config.get("image_size"),
                "num_classes": model_config.get("num_classes"),
                "classes": class_names,
                "best_phase": best_phase,
                "best_val_loss": best_val_loss,
                "checkpoint": final_path.name,
                "training": training,
                "finetuning": finetuning,
                "epochs_trained": {
                    "phase1": len(history1.history["loss"]),
                    "phase2": (
                        len(history2.history["loss"]) if history2 else 0
                    ),
                },
                "history": merged_history,
            },
            "model_card.json",
            artifact_path="model",
        )

        # Register the retained model; fall back to the raw checkpoint as a
        # plain artifact when the PyFunc wrapper cannot be built.
        model_logged = tracker.log_keras_checkpoint(
            final_path,
            class_names,
            image_size=tuple(model_config.get("image_size", (299, 299))),
            registered_model_name=mlflow_config.get("registered_model_name"),
        )
        if not model_logged:
            tracker.log_artifacts(final_path, artifact_path="model_file")

        print(
            f"\nBest model (phase {best_phase}, "
            f"val_loss={best_val_loss:.4f}) saved to: {final_path}"
        )
        print(f"Training history saved to: {history_path}")
        if tracker.run_id:
            print(f"MLflow run id: {tracker.run_id}")


if __name__ == "__main__":
    main()
