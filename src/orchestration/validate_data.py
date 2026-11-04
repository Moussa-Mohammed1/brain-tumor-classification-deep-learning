"""Validate the structure and quality of the classification and detection datasets.

Checks performed
----------------
Classification (``data/classification_split``):
  * the Train / Validation / Test splits exist,
  * every split contains the same non-empty class folders,
  * image counts per class are reported,
  * a sample of images is opened with Pillow to detect corrupted files.

Detection (``data/detection/data.yaml``):
  * ``path``, ``train``, ``val``, ``test``, ``nc`` and ``names`` are coherent,
  * every image has a matching ``.txt`` label and vice versa,
  * labels follow the YOLO format (5 fields, class id in range, normalised
    coordinates, positive width/height),
  * per-class box counts are reported,
  * a sample of images is verified with Pillow.

The JSON report is written to ``reports/orchestration/data_validation.json``
and the command exits with status 1 when blocking errors are found.

Usage:
    python -m src.orchestration.validate_data
    python -m src.orchestration.validate_data --image-sample 0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASSIFICATION_SPLITS = ("Train", "Validation", "Test")
DETECTION_SPLITS = ("train", "valid", "test")
DETECTION_SPLIT_KEYS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _label_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() == ".txt"
    }


def _verify_images(paths: list[Path], limit: int) -> tuple[int, list[str]]:
    """Open up to ``limit`` images with Pillow and report corrupted files."""
    if limit <= 0:
        return 0, []

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is a project dependency
        return 0, []

    checked = 0
    errors = []
    for path in paths[:limit]:
        checked += 1
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:  # noqa: BLE001 - any decoder error is relevant
            errors.append(f"{path}: {exc}")
    return checked, errors


def validate_classification(
    root: Path,
    image_sample: int = 5,
) -> dict:
    """Validate ``data/classification_split`` and return a report dict."""
    report = {
        "root": str(root),
        "splits": {},
        "errors": [],
        "warnings": [],
        "images_checked": 0,
    }

    if not root.is_dir():
        report["errors"].append(f"Classification directory not found: {root}")
        return report

    class_sets = {}

    for split in CLASSIFICATION_SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            report["errors"].append(f"Missing split: {split_dir}")
            continue

        class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        if not class_dirs:
            report["errors"].append(f"No class folders in {split_dir}")

        split_info = {
            "path": str(split_dir),
            "classes": {},
            "total_images": 0,
        }

        for class_dir in class_dirs:
            images = _image_files(class_dir)
            split_info["classes"][class_dir.name] = len(images)
            split_info["total_images"] += len(images)

            if not images:
                report["errors"].append(
                    f"No images for class '{class_dir.name}' in {split}"
                )
                continue

            checked, image_errors = _verify_images(images, image_sample)
            report["images_checked"] += checked
            report["errors"].extend(image_errors)

        class_sets[split] = set(split_info["classes"])
        report["splits"][split] = split_info

    if class_sets:
        reference = next(iter(class_sets.values()))
        for split, classes in class_sets.items():
            if classes != reference:
                report["warnings"].append(
                    f"Class folders differ in '{split}': "
                    f"{sorted(classes)} vs {sorted(reference)}"
                )

    return report


def _resolve_split_dir(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def _parse_label_file(
    label_path: Path,
    num_classes: int,
) -> tuple[dict, int, list[str], bool]:
    """Parse one YOLO label file.

    Returns ``(boxes_per_class, invalid_lines, errors, is_empty)``.
    """
    boxes_per_class: dict[int, int] = {}
    errors: list[str] = []
    invalid_lines = 0

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes_per_class, invalid_lines, errors, True

    for line_number, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            errors.append(
                f"{label_path}: line {line_number}: expected 5 fields, "
                f"got {len(parts)}"
            )
            invalid_lines += 1
            continue

        try:
            class_id = int(parts[0])
            cx, cy, width, height = (float(value) for value in parts[1:])
        except ValueError:
            errors.append(
                f"{label_path}: line {line_number}: non-numeric value"
            )
            invalid_lines += 1
            continue

        problems = []
        if not 0 <= class_id < num_classes:
            problems.append(
                f"class id {class_id} outside [0, {num_classes - 1}]"
            )
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
            problems.append("coordinates outside [0, 1]")
        if width <= 0 or height <= 0:
            problems.append("non-positive width/height")

        if problems:
            errors.append(
                f"{label_path}: line {line_number}: " + "; ".join(problems)
            )
            invalid_lines += 1
            continue

        boxes_per_class[class_id] = boxes_per_class.get(class_id, 0) + 1

    return boxes_per_class, invalid_lines, errors, False


def validate_detection(
    data_yaml_path: Path,
    image_sample: int = 5,
) -> dict:
    """Validate the YOLO dataset described by ``data.yaml``."""
    report = {
        "data_yaml": str(data_yaml_path),
        "splits": {},
        "errors": [],
        "warnings": [],
        "images_checked": 0,
    }

    if not data_yaml_path.is_file():
        report["errors"].append(f"data.yaml not found: {data_yaml_path}")
        return report

    try:
        config = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        report["errors"].append(f"Invalid YAML in {data_yaml_path}: {exc}")
        return report

    names = config.get("names", [])
    if isinstance(names, dict):
        names = [
            name for _, name in sorted(names.items(), key=lambda item: int(item[0]))
        ]
    names = list(names)

    num_classes = int(config.get("nc", len(names)))
    if not names:
        report["errors"].append("data.yaml has no 'names' entry")
    if num_classes != len(names):
        report["errors"].append(
            f"'nc' ({num_classes}) does not match the number of names "
            f"({len(names)})"
        )

    base = Path(config.get("path") or data_yaml_path.parent)
    if not base.is_absolute():
        base = (data_yaml_path.parent / base).resolve()
    report["path"] = str(base)

    for split in DETECTION_SPLITS:
        split_key = DETECTION_SPLIT_KEYS[split]
        if split_key not in config:
            report["warnings"].append(
                f"Split '{split}' (key '{split_key}') not declared in data.yaml"
            )
            continue

        images_dir = _resolve_split_dir(base, config[split_key])
        labels_dir = images_dir.parent / "labels"
        if images_dir.name != "images":
            labels_dir = images_dir / "labels"

        split_info = {
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "images": 0,
            "labels": 0,
            "boxes": 0,
            "empty_labels": 0,
            "per_class": {},
        }

        if not images_dir.is_dir():
            report["errors"].append(f"Missing images directory: {images_dir}")
            report["splits"][split] = split_info
            continue
        if not labels_dir.is_dir():
            report["errors"].append(f"Missing labels directory: {labels_dir}")
            report["splits"][split] = split_info
            continue

        images = {
            path.stem: path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        labels = _label_files(labels_dir)

        split_info["images"] = len(images)
        split_info["labels"] = len(labels)

        missing_labels = sorted(set(images) - set(labels))
        orphan_labels = sorted(set(labels) - set(images))

        if missing_labels:
            report["errors"].append(
                f"{split}: {len(missing_labels)} image(s) without annotation "
                f"(e.g. {missing_labels[:5]})"
            )
        if orphan_labels:
            report["errors"].append(
                f"{split}: {len(orphan_labels)} annotation(s) without image "
                f"(e.g. {orphan_labels[:5]})"
            )

        per_class = {name: 0 for name in names}
        for label_path in labels.values():
            boxes, invalid, errors, is_empty = _parse_label_file(
                label_path, num_classes
            )
            report["errors"].extend(errors)
            split_info["boxes"] += sum(boxes.values())
            if is_empty:
                split_info["empty_labels"] += 1
            for class_id, count in boxes.items():
                if 0 <= class_id < len(names):
                    per_class[names[class_id]] += count

        split_info["per_class"] = per_class

        if split_info["empty_labels"]:
            report["warnings"].append(
                f"{split}: {split_info['empty_labels']} empty label file(s) "
                "(negative samples)"
            )

        checked, image_errors = _verify_images(
            sorted(images.values()), image_sample
        )
        report["images_checked"] += checked
        report["errors"].extend(image_errors)

        report["splits"][split] = split_info

    return report


def validate_all(
    classification_dir: Path,
    detection_yaml: Path,
    image_sample: int = 5,
) -> dict:
    """Run both validations and return the combined report."""
    classification = validate_classification(
        classification_dir, image_sample=image_sample
    )
    detection = validate_detection(
        detection_yaml, image_sample=image_sample
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": not (classification["errors"] or detection["errors"]),
        "classification": classification,
        "detection": detection,
    }


def print_summary(report: dict) -> None:
    print("=== Dataset validation ===")

    print("\nClassification:")
    for split, info in report["classification"]["splits"].items():
        classes = ", ".join(
            f"{name}={count}" for name, count in info["classes"].items()
        )
        print(f"  {split:10s} total={info['total_images']:5d} | {classes}")

    print("\nDetection:")
    for split, info in report["detection"]["splits"].items():
        classes = ", ".join(
            f"{name}={count}" for name, count in info["per_class"].items()
        )
        print(
            f"  {split:5s} images={info['images']:5d} "
            f"labels={info['labels']:5d} boxes={info['boxes']:5d} | {classes}"
        )

    errors = (
        report["classification"]["errors"] + report["detection"]["errors"]
    )
    warnings = (
        report["classification"]["warnings"]
        + report["detection"]["warnings"]
    )

    print(f"\nStatus: {'OK' if report['ok'] else 'FAILED'}")
    print(f"Errors: {len(errors)} | Warnings: {len(warnings)}")
    for error in errors[:20]:
        print(f"  [ERROR] {error}")
    if len(errors) > 20:
        print(f"  ... and {len(errors) - 20} more errors")
    for warning in warnings[:10]:
        print(f"  [WARN]  {warning}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate the classification and detection datasets."
    )
    parser.add_argument(
        "--classification-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "classification_split",
    )
    parser.add_argument(
        "--detection-yaml",
        type=Path,
        default=PROJECT_ROOT / "data" / "detection" / "data.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "orchestration"
        / "data_validation.json",
    )
    parser.add_argument(
        "--image-sample",
        type=int,
        default=5,
        help="Images opened per class/split to detect corruption (0 disables).",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Always exit 0 (useful for exploratory runs).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    report = validate_all(
        args.classification_dir,
        args.detection_yaml,
        image_sample=args.image_sample,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, default=str)

    print_summary(report)
    print(f"\nReport written to: {args.output}")

    if report["ok"] or args.allow_failures:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
