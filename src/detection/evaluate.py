"""Evaluate the trained YOLOv8 detector on the detection splits.

Prerequisite:
    python -m src.detection.train   # produces models/detection/best.pt

Usage:
    python -m src.detection.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
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


def resolve_names(names) -> dict:
    """Return class names as ``{class_id: name}`` for dict or list input."""
    if isinstance(names, dict):
        return {int(key): value for key, value in names.items()}
    return {index: value for index, value in enumerate(names)}


def metrics_to_dict(metrics) -> dict:
    """Convert an Ultralytics DetMetrics object into a JSON-friendly dict."""
    names = resolve_names(metrics.names)
    box = metrics.box

    result = {
        "overall": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
        },
        "per_class": {},
    }

    ap_class_index = getattr(box, "ap_class_index", [])
    for position, class_id in enumerate(ap_class_index):
        class_id = int(class_id)
        result["per_class"][names.get(class_id, str(class_id))] = {
            "precision": float(box.p[position]),
            "recall": float(box.r[position]),
            "mAP50": float(box.ap50[position]),
            "mAP50-95": float(box.ap[position]),
        }

    result["speed_ms"] = {
        key: float(value)
        for key, value in getattr(metrics, "speed", {}).items()
    }
    return result


def evaluate_split(
    model,
    data_yaml: Path,
    split: str,
    eval_config: dict,
    run_name: str,
) -> dict:
    val_args = {
        "data": str(data_yaml),
        "split": split,
        "imgsz": eval_config.get("imgsz", 640),
        "batch": eval_config.get("batch", 16),
        "conf": eval_config.get("conf", 0.001),
        "iou": eval_config.get("iou", 0.6),
        "plots": eval_config.get("plots", True),
        "project": str(REPORT_DIR / "runs"),
        "name": f"{run_name}_{split}",
        "exist_ok": True,
    }
    device = eval_config.get("device")
    if device is not None:
        val_args["device"] = device

    metrics = model.val(**val_args)
    return metrics_to_dict(metrics)


COLORS = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 128, 255),
    (255, 0, 255),
]


def save_ground_truth_samples(class_names, eval_config: dict):
    """Draw ground-truth boxes on a few test images (context step: visual check)."""
    count = int(eval_config.get("save_samples", 0))
    if count <= 0:
        return None

    images_dir = DATA_YAML.parent / "test" / "images"
    labels_dir = DATA_YAML.parent / "test" / "labels"
    output_dir = REPORT_DIR / "samples_ground_truth"
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(images_dir.iterdir())[:count]:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width = image.shape[:2]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            cx, cy, box_w, box_h = (float(value) for value in parts[1:])
            x1 = int((cx - box_w / 2) * width)
            y1 = int((cy - box_h / 2) * height)
            x2 = int((cx + box_w / 2) * width)
            y2 = int((cy + box_h / 2) * height)

            color = COLORS[class_id % len(COLORS)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = class_names[class_id] if class_id < len(class_names) else str(class_id)
            cv2.putText(
                image,
                label,
                (x1, max(y1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.imwrite(str(output_dir / image_path.name), image)

    return output_dir


def save_predictions(model, eval_config: dict, run_name: str):
    """Render predictions for a few test images (qualitative check)."""
    count = int(eval_config.get("save_predictions", 0))
    if count <= 0:
        return None

    images = sorted((DATA_YAML.parent / "test" / "images").glob("*"))[:count]
    if not images:
        return None

    project = REPORT_DIR / "predictions"
    predict_args = {
        "source": [str(path) for path in images],
        "conf": eval_config.get("predict_conf", 0.25),
        "save": True,
        "project": str(project),
        "name": run_name,
        "exist_ok": True,
    }
    device = eval_config.get("device")
    if device is not None:
        predict_args["device"] = device

    model.predict(**predict_args)
    return project / run_name


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"{DATA_YAML} not found. "
            "Run `python -m src.data.prepare_detection` first."
        )

    weights_path = MODEL_DIR / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"{weights_path} not found. "
            "Run `python -m src.detection.train` first."
        )

    config = load_config()
    eval_config = config.get("evaluation", {})
    splits = eval_config.get("splits", ["val", "test"])
    run_name = config.get("training", {}).get("name", "yolov8n_brain_tumor")

    model = YOLO(str(weights_path))

    all_metrics = {}
    for split in splits:
        print(f"\n=== Evaluation on '{split}' split ===")
        metrics = evaluate_split(
            model, DATA_YAML, split, eval_config, run_name
        )
        all_metrics[split] = metrics

        overall = metrics["overall"]
        print(f"Precision:    {overall['precision']:.4f}")
        print(f"Recall:       {overall['recall']:.4f}")
        print(f"mAP@0.5:      {overall['mAP50']:.4f}")
        print(f"mAP@0.5:0.95: {overall['mAP50-95']:.4f}")

        for class_name, values in metrics["per_class"].items():
            print(
                f"  {class_name:12s} "
                f"P={values['precision']:.3f} "
                f"R={values['recall']:.3f} "
                f"mAP50={values['mAP50']:.3f} "
                f"mAP50-95={values['mAP50-95']:.3f}"
            )

    metrics_path = REPORT_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=4)

    class_names = config.get("data", {}).get("classes", [])
    sample_dir = save_ground_truth_samples(class_names, eval_config)
    prediction_dir = save_predictions(model, eval_config, run_name)

    print(f"\nMetrics saved to: {metrics_path}")
    if sample_dir is not None:
        print(f"Ground-truth samples saved to: {sample_dir}")
    if prediction_dir is not None:
        print(f"Prediction images saved to: {prediction_dir}")


if __name__ == "__main__":
    main()
