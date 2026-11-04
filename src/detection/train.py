"""Train a YOLOv8 detector for brain-tumor detection.

Prerequisite:
    python -m src.data.prepare_detection   # builds data/detection/

Usage:
    python -m src.detection.train

Every run is tracked in MLflow (hyperparameters, per-epoch curves, final
metrics, plots and the retained ``best.pt``). Set ``MLFLOW_TRACKING_URI`` to use
a remote server; otherwise runs are stored in ``mlruns/``.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

from src.tracking.mlflow_utils import MLflowTracker, utc_timestamp


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "configs" / "detection.yaml"
DATA_YAML = ROOT_DIR / "data" / "detection" / "data.yaml"
MODEL_DIR = ROOT_DIR / "models" / "detection"
REPORT_DIR = ROOT_DIR / "reports" / "detection"

TRAINING_PLOTS = (
    "results.csv",
    "results.png",
    "PR_curve.png",
    "P_curve.png",
    "R_curve.png",
    "F1_curve.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "labels.jpg",
    "args.yaml",
)

METRIC_ALIASES = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
    "fitness": "fitness",
}


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize_results(results) -> dict:
    """Extract a JSON-friendly metrics dict from Ultralytics results."""
    if hasattr(results, "results_dict"):
        return {
            key: float(value)
            for key, value in results.results_dict.items()
        }
    if isinstance(results, dict):
        return {key: float(value) for key, value in results.items()}
    return {}


def dataset_counts() -> dict:
    """Count images per detection split."""
    counts = {}
    for split in ("train", "valid", "test"):
        images_dir = DATA_YAML.parent / split / "images"
        counts[split] = (
            sum(1 for path in images_dir.iterdir() if path.is_file())
            if images_dir.is_dir()
            else 0
        )
    return counts


def log_training_curve(tracker: MLflowTracker, results_csv: Path) -> None:
    """Log Ultralytics' per-epoch ``results.csv`` as MLflow metric series."""
    if not results_csv.exists():
        return

    with open(results_csv, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for epoch, row in enumerate(reader):
            values = {}
            for key, value in row.items():
                if key is None:
                    continue
                try:
                    values[key.strip()] = float(value)
                except (TypeError, ValueError):
                    continue
            tracker.log_metrics(values, prefix="curve/", step=epoch)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"{DATA_YAML} not found. "
            "Run `python -m src.data.prepare_detection` first."
        )

    config = load_config()
    training = dict(config["training"])
    augmentation = training.pop("augmentation", {})
    classes = config.get("data", {}).get("classes", [])
    mlflow_config = config.get("mlflow", {})

    model_name = training.pop("model", "yolov8n.pt")
    project = (ROOT_DIR / training.pop("project", "models/detection")).resolve()
    run_name = training.pop("name", "yolov8n_brain_tumor")

    # `device: null` means "let Ultralytics choose" (CPU or first GPU).
    if training.get("device") is None:
        training.pop("device", None)

    train_args = {**training, **augmentation}

    print("=== YOLOv8 detection training ===")
    print(f"Model:   {model_name}")
    print(f"Data:    {DATA_YAML}")
    print(f"Project: {project}")
    print(f"Run:     {run_name}")
    print(f"Args:    {train_args}")

    with MLflowTracker(
        mlflow_config,
        default_experiment="brainscan-detection",
        run_name=f"detection-{utc_timestamp()}",
        tags={
            "stage": "training",
            "task": "detection",
            "model": model_name,
        },
    ) as tracker:
        tracker.log_params(
            {
                "model": model_name,
                "data_yaml": str(DATA_YAML),
                "training": {**training, "augmentation": augmentation},
                "data": {
                    "classes": classes,
                    "num_classes": len(classes),
                    "images": dataset_counts(),
                },
            }
        )
        tracker.log_artifacts(CONFIG_PATH, artifact_path="config")
        tracker.log_artifacts(DATA_YAML, artifact_path="config")

        model = YOLO(model_name)
        results = model.train(
            data=str(DATA_YAML),
            project=str(project),
            name=run_name,
            exist_ok=True,
            **train_args,
        )

        save_dir = Path(model.trainer.save_dir)
        best_weights = save_dir / "weights" / "best.pt"
        last_weights = save_dir / "weights" / "last.pt"

        final_weights = MODEL_DIR / "best.pt"
        source_weights = (
            best_weights if best_weights.exists() else last_weights
        )
        if source_weights.exists():
            shutil.copy2(source_weights, final_weights)

        # Keep everything needed to reuse / deploy the model.
        run_config = {
            "model": model_name,
            "data": str(DATA_YAML),
            "project": str(project),
            "name": run_name,
            "training": {**training, "augmentation": augmentation},
        }
        training_config_path = MODEL_DIR / "training_config.yaml"
        with open(training_config_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(run_config, file, sort_keys=False)

        classes_path = MODEL_DIR / "classes.json"
        with open(classes_path, "w", encoding="utf-8") as file:
            json.dump(classes, file, indent=4)

        metrics = normalize_results(results)
        metrics_path = REPORT_DIR / "training_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        # MLflow: metrics, curves and artifacts.
        tracker.log_metrics(
            {
                METRIC_ALIASES.get(key, key.replace("metrics/", "")): value
                for key, value in metrics.items()
            }
        )
        tracker.log_metrics(
            {"best/fitness": metrics.get("fitness", 0.0)}
        )
        log_training_curve(tracker, save_dir / "results.csv")

        tracker.log_artifacts(
            [save_dir / name for name in TRAINING_PLOTS],
            artifact_path="training",
        )
        tracker.log_artifacts(
            [metrics_path, training_config_path, classes_path],
            artifact_path="reports",
        )
        tracker.log_dict(
            {
                "model": model_name,
                "data_yaml": str(DATA_YAML),
                "classes": classes,
                "num_classes": len(classes),
                "images": dataset_counts(),
                "training": {**training, "augmentation": augmentation},
                "metrics": metrics,
                "weights": final_weights.name,
                "ultralytics_run": str(save_dir),
            },
            "model_card.json",
            artifact_path="model",
        )

        tracker.set_tags(
            {
                "classes": ",".join(str(name) for name in classes),
                "ultralytics_run": str(save_dir),
            }
        )

        if final_weights.exists():
            tracker.log_yolo_model(
                final_weights,
                classes_path,
                registered_model_name=mlflow_config.get(
                    "registered_model_name"
                ),
                conf=config.get("evaluation", {}).get("predict_conf", 0.25),
            )

        print("\n=== Training finished ===")
        print(f"Run directory: {save_dir}")
        print(f"Best weights:  {final_weights}")
        print(f"Metrics:       {metrics_path}")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        if tracker.run_id:
            print(f"MLflow run id: {tracker.run_id}")


if __name__ == "__main__":
    main()
