"""Train a YOLOv8 detector for brain-tumor detection.

Prerequisite:
    python -m src.data.prepare_detection   # builds data/detection/

Usage:
    python -m src.detection.train
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "configs" / "detection.yaml"
DATA_YAML = ROOT_DIR / "data" / "detection" / "data.yaml"
MODEL_DIR = ROOT_DIR / "models" / "detection"
REPORT_DIR = ROOT_DIR / "reports" / "detection"


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
    source_weights = best_weights if best_weights.exists() else last_weights
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
    with open(MODEL_DIR / "training_config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(run_config, file, sort_keys=False)

    classes = config.get("data", {}).get("classes", [])
    with open(MODEL_DIR / "classes.json", "w", encoding="utf-8") as file:
        json.dump(classes, file, indent=4)

    metrics = normalize_results(results)
    metrics_path = REPORT_DIR / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n=== Training finished ===")
    print(f"Run directory: {save_dir}")
    print(f"Best weights:  {final_weights}")
    print(f"Metrics:       {metrics_path}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
