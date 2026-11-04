import json
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

from src.orchestration.artifacts import archive_run, sanitize_run_id
from src.orchestration.checks import check_environment
from src.orchestration.validate_data import (
    validate_classification,
    validate_detection,
)


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(path)


def _make_classification_tree(root: Path, classes=("Glioma", "Pituitary")):
    for split in ("Train", "Validation", "Test"):
        for class_name in classes:
            _make_image(root / split / class_name / "images" / "scan.jpg")


def _make_detection_tree(root: Path):
    names = ["Glioma", "Pituitary"]
    for split in ("train", "valid", "test"):
        images = root / split / "images"
        labels = root / split / "labels"
        _make_image(images / "a.jpg")
        _make_image(images / "b.jpg")
        labels.mkdir(parents=True, exist_ok=True)
        (labels / "a.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n", encoding="utf-8"
        )
        (labels / "b.txt").write_text(
            "0 0.4 0.4 0.3 0.3\n", encoding="utf-8"
        )

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "nc: 2",
                "names:",
                "- Glioma",
                "- Pituitary",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_validate_classification_ok(tmp_path: Path):
    _make_classification_tree(tmp_path)

    report = validate_classification(tmp_path, image_sample=1)

    assert report["errors"] == []
    assert report["splits"]["Train"]["total_images"] == 2
    assert report["splits"]["Train"]["classes"]["Glioma"] == 1


def test_validate_classification_detects_missing_split(tmp_path: Path):
    _make_classification_tree(tmp_path)
    shutil.rmtree(tmp_path / "Test")

    report = validate_classification(tmp_path, image_sample=0)

    assert any("Missing split" in error for error in report["errors"])


def test_validate_detection_ok(tmp_path: Path):
    data_yaml = _make_detection_tree(tmp_path)

    report = validate_detection(data_yaml, image_sample=1)

    assert report["errors"] == []
    assert report["splits"]["train"]["images"] == 2
    assert report["splits"]["train"]["labels"] == 2
    assert report["splits"]["train"]["boxes"] == 3
    assert report["splits"]["valid"]["images"] == 2


def test_validate_detection_detects_missing_label(tmp_path: Path):
    data_yaml = _make_detection_tree(tmp_path)
    (tmp_path / "train" / "labels" / "b.txt").unlink()

    report = validate_detection(data_yaml, image_sample=0)

    assert any("without annotation" in error for error in report["errors"])


def test_validate_detection_detects_invalid_box(tmp_path: Path):
    data_yaml = _make_detection_tree(tmp_path)
    (tmp_path / "train" / "labels" / "a.txt").write_text(
        "0 1.5 0.5 0.2 0.2\n", encoding="utf-8"
    )

    report = validate_detection(data_yaml, image_sample=0)

    assert any("coordinates outside" in error for error in report["errors"])


def _make_project(root: Path):
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "classification.yaml").write_text(
        "model:\n  name: test\n", encoding="utf-8"
    )
    (root / "reports" / "classification").mkdir(parents=True)
    (root / "reports" / "classification" / "evaluation_metrics.json").write_text(
        json.dumps({"Test": {"accuracy": 0.9}}), encoding="utf-8"
    )
    (root / "models" / "detection").mkdir(parents=True)
    (root / "models" / "detection" / "best.pt").write_bytes(b"fake-weights")


def test_archive_run_copies_files_and_writes_manifest(tmp_path: Path):
    project = tmp_path / "project"
    _make_project(project)

    manifest = archive_run(
        run_id="manual__2026-11-04T10:00:00+00:00",
        output_root=tmp_path / "out",
        copy_models=True,
        project_root=project,
    )

    run_dir = Path(manifest["manifest_path"]).parent
    assert run_dir.is_dir()
    assert (run_dir / "configs" / "classification.yaml").is_file()
    assert (run_dir / "models" / "detection" / "best.pt").is_file()
    assert manifest["metrics"]["classification"]["test"]["accuracy"] == 0.9

    model = manifest["models"][0]
    assert len(model["sha256"]) == 64
    assert "archived" in model

    with open(run_dir / "manifest.json", encoding="utf-8") as file:
        saved = json.load(file)
    assert saved["run_id"] == "manual__2026-11-04T10:00:00+00:00"


def test_archive_run_can_skip_model_copies(tmp_path: Path):
    project = tmp_path / "project"
    _make_project(project)

    manifest = archive_run(
        run_id="run-2",
        output_root=tmp_path / "out",
        copy_models=False,
        project_root=project,
    )

    assert manifest["copy_models"] is False
    assert "archived" not in manifest["models"][0]
    assert not (Path(manifest["manifest_path"]).parent / "models").exists()


def test_sanitize_run_id():
    assert sanitize_run_id("manual__2026-11-04T10:00:00+00:00") == (
        "manual__2026-11-04T10_00_00_00_00"
    )


def test_check_environment_ok(tmp_path: Path):
    project = tmp_path / "project"
    _make_project(project)
    (project / "configs" / "detection.yaml").write_text(
        "data: {}\n", encoding="utf-8"
    )
    for relative in (
        "src/classification/train.py",
        "src/classification/evaluate.py",
        "src/detection/train.py",
        "src/detection/evaluate.py",
        "src/data/prepare_detection.py",
        "src/orchestration/validate_data.py",
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    (project / "data" / "classification" / "Train").mkdir(parents=True)
    _make_classification_tree(project / "data" / "classification_split")

    summary = check_environment(
        python_executable=sys.executable,
        verify_dependencies=False,
        project_root=project,
    )

    assert summary["ok"] is True
    assert summary["classification"]["images"]["Train"] == 2
    assert summary["detection"]["status"].startswith("missing")


def test_check_environment_reports_missing_files(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()

    with pytest.raises(RuntimeError) as error:
        check_environment(
            python_executable=sys.executable,
            verify_dependencies=False,
            project_root=project,
        )

    message = str(error.value)
    assert "Missing config files" in message
    assert "Missing raw classification data" in message
