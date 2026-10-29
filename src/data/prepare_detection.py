"""Prepare the YOLOv8 detection dataset from the raw Kaggle MRI data.

Raw layout (as downloaded):
    data/classification/{Train,Val}/{class}/images/*.{jpg,jpeg,png,bmp}
    data/classification/{Train,Val}/{class}/labels/*.txt   (YOLO format)

Output layout:
    data/detection/train/images/*
    data/detection/train/labels/*
    data/detection/valid/images/*
    data/detection/valid/labels/*
    data/detection/test/images/*
    data/detection/test/labels/*
    data/detection/data.yaml

The script:
  * matches every image with its ``.txt`` annotation by base filename,
  * rejects images without annotation and annotations without image,
  * validates YOLO boxes (5 fields, class id in range, coords in [0, 1]),
  * checks that each class folder maps to the expected YOLO class id,
  * builds the train / valid / test splits (test is carved out of Val when
    the source dataset has no Test folder),
  * writes ``data.yaml`` for Ultralytics YOLOv8.

Usage:
    python -m src.data.prepare_detection
    python -m src.data.prepare_detection --overwrite
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "configs" / "detection.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
OUTPUT_SPLITS = ("train", "valid", "test")


@dataclass
class Sample:
    image: Path
    label: Path
    source_split: str
    class_name: str
    boxes: list = field(default_factory=list)


def parse_label_file(label_path: Path, num_classes: int):
    """Parse a YOLO label file and return ``(boxes, errors)``.

    Each line must contain exactly 5 values: ``class_id cx cy w h``.
    Coordinates must be normalised to [0, 1] and width/height positive.
    """
    boxes = []
    errors = []

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes, ["empty label file"]

    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(
                f"line {line_no}: expected 5 fields, got {len(parts)}"
            )
            continue

        try:
            class_id = int(parts[0])
            cx, cy, width, height = (float(value) for value in parts[1:])
        except ValueError:
            errors.append(f"line {line_no}: non-numeric value")
            continue

        line_errors = []
        if not 0 <= class_id < num_classes:
            line_errors.append(
                f"line {line_no}: class id {class_id} outside "
                f"[0, {num_classes - 1}]"
            )
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
            line_errors.append(
                f"line {line_no}: coordinates outside [0, 1]"
            )
        if width <= 0 or height <= 0:
            line_errors.append(
                f"line {line_no}: non-positive width/height"
            )

        if line_errors:
            errors.extend(line_errors)
            continue

        boxes.append((class_id, cx, cy, width, height))

    return boxes, errors


def collect_samples(
    source_dir: Path,
    source_splits,
    class_names,
    keep_empty_labels: bool,
):
    """Walk the raw dataset and return valid samples plus rejection reports."""
    samples = []
    rejected = []
    orphan_labels = []
    num_classes = len(class_names)

    for split_name in source_splits:
        split_dir = source_dir / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Source split not found: {split_dir}")

        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            class_name = class_dir.name
            if class_name not in class_names:
                raise ValueError(
                    f"Unknown class folder '{class_name}'. "
                    f"Expected one of {class_names}."
                )

            images_dir = class_dir / "images"
            labels_dir = class_dir / "labels"
            if not images_dir.is_dir() or not labels_dir.is_dir():
                continue

            image_paths = {
                path.stem: path
                for path in images_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            }
            label_paths = {
                path.stem: path
                for path in labels_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".txt"
            }

            # Annotations without a matching image.
            for stem, label_path in label_paths.items():
                if stem not in image_paths:
                    orphan_labels.append(label_path)

            for stem, image_path in image_paths.items():
                label_path = label_paths.get(stem)
                if label_path is None:
                    rejected.append((image_path, "missing label file"))
                    continue

                boxes, errors = parse_label_file(label_path, num_classes)
                real_errors = [
                    error for error in errors if error != "empty label file"
                ]
                if real_errors:
                    rejected.append((label_path, "; ".join(real_errors)))
                    continue
                if not boxes and not keep_empty_labels:
                    rejected.append((label_path, "empty label file"))
                    continue

                samples.append(
                    Sample(
                        image=image_path,
                        label=label_path,
                        source_split=split_name,
                        class_name=class_name,
                        boxes=boxes,
                    )
                )

    return samples, rejected, orphan_labels


def validate_class_mapping(samples, class_names):
    """Check that each class folder matches its index in ``class_names``."""
    ids_by_class = defaultdict(Counter)
    for sample in samples:
        for class_id, _, _, _, _ in sample.boxes:
            ids_by_class[sample.class_name][class_id] += 1

    mapping = {}
    for class_name, counter in ids_by_class.items():
        if not counter:
            continue
        dominant_id, _ = counter.most_common(1)[0]
        expected_id = class_names.index(class_name)
        if dominant_id != expected_id:
            raise ValueError(
                f"Class coherence error: folder '{class_name}' uses YOLO id "
                f"{dominant_id} but is index {expected_id} in {class_names}."
            )
        mapping[dominant_id] = class_name

    return mapping


def stratified_split(samples, test_ratio: float, seed: int):
    """Split ``samples`` by class into valid/test sets."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.class_name].append(sample)

    valid = []
    test = []

    for class_name in sorted(groups):
        items = groups[class_name]
        rng.shuffle(items)

        if len(items) <= 1 or test_ratio <= 0:
            valid.extend(items)
            continue

        n_test = max(1, round(len(items) * test_ratio))
        n_test = min(n_test, len(items) - 1)
        test.extend(items[:n_test])
        valid.extend(items[n_test:])

    return valid, test


def copy_samples(samples, split_dir: Path, class_names):
    """Copy image/label pairs into ``<split_dir>/images`` and ``labels``."""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    used_names = set()

    for sample in samples:
        target_image = images_dir / sample.image.name
        target_label = labels_dir / f"{sample.image.stem}.txt"
        prefix = sample.class_name.replace(" ", "_")

        counter = 1
        while target_image.name in used_names or target_image.exists():
            target_image = images_dir / f"{prefix}_{counter}_{sample.image.name}"
            target_label = labels_dir / (
                f"{prefix}_{counter}_{sample.image.stem}.txt"
            )
            counter += 1

        used_names.add(target_image.name)
        shutil.copy2(sample.image, target_image)
        shutil.copy2(sample.label, target_label)


def write_data_yaml(output_dir: Path, class_names) -> Path:
    data = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": list(class_names),
    }
    data_yaml_path = output_dir / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)
    return data_yaml_path


def summarize(split_name: str, samples, class_names):
    boxes_by_class = Counter()
    for sample in samples:
        for box in sample.boxes:
            boxes_by_class[class_names[box[0]]] += 1

    per_class = ", ".join(
        f"{name}={boxes_by_class[name]}" for name in class_names
    )
    print(
        f"  {split_name:5s} images={len(samples):5d} "
        f"boxes={sum(boxes_by_class.values()):5d} | {per_class}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build the YOLOv8 dataset in data/detection/."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to configs/detection.yaml.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild data/detection/ if it already exists.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_config = config["data"]
    class_names = list(data_config["classes"])

    source_dir = (ROOT_DIR / data_config["source_dir"]).resolve()
    output_dir = (ROOT_DIR / data_config["output_dir"]).resolve()

    train_split = data_config["train_split"]
    val_split = data_config["val_split"]
    test_split = data_config.get("test_split")
    keep_empty_labels = bool(data_config.get("keep_empty_labels", False))
    seed = int(data_config.get("seed", 42))
    val_test_ratio = float(data_config.get("val_test_ratio", 0.5))
    overwrite = bool(args.overwrite or data_config.get("overwrite", False))

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise SystemExit(
            f"{output_dir} already exists and is not empty. "
            "Re-run with --overwrite to rebuild it."
        )

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_splits = [train_split, val_split]
    if test_split:
        source_splits.append(test_split)

    print(f"Scanning {source_dir} ...")
    samples, rejected, orphan_labels = collect_samples(
        source_dir, source_splits, class_names, keep_empty_labels
    )
    mapping = validate_class_mapping(samples, class_names)

    by_source = {
        split: [sample for sample in samples if sample.source_split == split]
        for split in source_splits
    }

    train_samples = by_source[train_split]
    if test_split:
        valid_samples = by_source[val_split]
        test_samples = by_source[test_split]
    else:
        valid_samples, test_samples = stratified_split(
            by_source[val_split], val_test_ratio, seed
        )

    split_samples = {
        "train": train_samples,
        "valid": valid_samples,
        "test": test_samples,
    }

    print("\nCopying files ...")
    for split_name, items in split_samples.items():
        copy_samples(items, output_dir / split_name, class_names)

    data_yaml_path = write_data_yaml(output_dir, class_names)

    print("\n=== Detection dataset ready ===")
    print(f"Source:  {source_dir}")
    print(f"Output:  {output_dir}")
    print(f"Classes: {class_names}")
    print(f"Mapping: {mapping}")
    print("\nPer split:")
    for split_name, items in split_samples.items():
        summarize(split_name, items, class_names)

    print(f"\nRejected files: {len(rejected)}")
    for path, reason in rejected[:10]:
        print(f"  - {path}: {reason}")
    if len(rejected) > 10:
        print(f"  ... and {len(rejected) - 10} more")

    print(f"\nOrphan label files: {len(orphan_labels)}")
    for path in orphan_labels[:10]:
        print(f"  - {path}")
    if len(orphan_labels) > 10:
        print(f"  ... and {len(orphan_labels) - 10} more")

    print(f"\ndata.yaml written to: {data_yaml_path}")


if __name__ == "__main__":
    main()
