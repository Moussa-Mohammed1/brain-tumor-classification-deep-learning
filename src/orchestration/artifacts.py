"""Archive the models, reports and configurations produced by a pipeline run.

The Airflow DAG calls :func:`archive_run` as its final data-producing step. It
creates a self-contained directory::

    artifacts/airflow_runs/<run_id>/
        manifest.json
        configs/classification.yaml
        configs/detection.yaml
        data/data.yaml
        models/classification/best_model.keras
        models/detection/best.pt
        reports/classification/...
        reports/detection/...
        reports/orchestration/...

Every archived file is hashed (SHA-256) and described in ``manifest.json``
together with the git commit, the metrics and the Python version, so a run can
be audited or reproduced later.

Usage:
    python -m src.orchestration.artifacts --run-id manual__2026-11-04
    python -m src.orchestration.artifacts --run-id test --no-models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "airflow_runs"

CONFIG_ARTIFACTS = (
    "configs/classification.yaml",
    "configs/detection.yaml",
    "data/detection/data.yaml",
    "requirements.txt",
)

REPORT_DIRS = (
    "reports/classification",
    "reports/detection",
    "reports/orchestration",
)

MODEL_ARTIFACTS = (
    "models/classification/best_model.keras",
    "models/classification/best_model_phase1.keras",
    "models/classification/best_model_phase2.keras",
    "models/detection/best.pt",
    "models/detection/classes.json",
    "models/detection/training_config.yaml",
)


def sanitize_run_id(run_id: str) -> str:
    """Make an Airflow run id safe for a directory name."""
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(run_id)
    )
    return safe.strip("._") or "run"


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


def collect_metrics(root: Path) -> dict:
    """Return a compact metrics snapshot for the manifest."""
    metrics = {"classification": {}, "detection": {}}

    classification = _read_json(
        root / "reports" / "classification" / "evaluation_metrics.json"
    )
    for split, values in classification.items():
        if not isinstance(values, dict):
            continue
        metrics["classification"][split.lower()] = {
            key: values.get(key)
            for key in (
                "loss",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
            )
            if key in values
        }

    detection = _read_json(
        root / "reports" / "detection" / "evaluation_metrics.json"
    )
    for split, values in detection.items():
        overall = values.get("overall", {}) if isinstance(values, dict) else {}
        metrics["detection"][split.lower()] = overall

    detection_training = _read_json(
        root / "reports" / "detection" / "training_metrics.json"
    )
    if detection_training:
        metrics["detection"]["training"] = detection_training

    return metrics


def _file_entry(path: Path, base: Path, archived_path: Path | None) -> dict:
    entry = {
        "source": str(path),
        "relative": str(path.relative_to(base)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256sum(path),
    }
    if archived_path is not None:
        entry["archived"] = str(archived_path)
    return entry


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def archive_run(
    run_id: str,
    execution_date=None,
    output_root: str | Path | None = None,
    copy_models: bool = True,
    project_root: str | Path | None = None,
) -> dict:
    """Archive configs, reports and models; return the manifest dict."""
    root = Path(project_root or PROJECT_ROOT).resolve()
    destination_root = Path(output_root or DEFAULT_OUTPUT_ROOT)
    if not destination_root.is_absolute():
        destination_root = root / destination_root

    run_dir = destination_root / sanitize_run_id(run_id)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    files: list[dict] = []

    # -- configs and dataset descriptor -----------------------------------
    for relative in CONFIG_ARTIFACTS:
        source = root / relative
        if not source.is_file():
            continue
        destination = run_dir / relative
        _copy_file(source, destination)
        files.append(_file_entry(source, root, destination.relative_to(run_dir)))

    # -- reports -----------------------------------------------------------
    for relative in REPORT_DIRS:
        source_dir = root / relative
        if not source_dir.is_dir():
            continue
        for source in sorted(source_dir.rglob("*")):
            if not source.is_file():
                continue
            relative_path = source.relative_to(root)
            destination = run_dir / relative_path
            _copy_file(source, destination)
            files.append(
                _file_entry(source, root, destination.relative_to(run_dir))
            )

    # -- models ------------------------------------------------------------
    models: list[dict] = []
    missing_models: list[str] = []
    seen = set()
    for relative in MODEL_ARTIFACTS:
        if relative in seen:
            continue
        seen.add(relative)

        source = root / relative
        if not source.is_file():
            missing_models.append(relative)
            continue

        archived_path = None
        if copy_models:
            destination = run_dir / relative
            _copy_file(source, destination)
            archived_path = destination

        models.append(_file_entry(source, root, archived_path))

    manifest = {
        "run_id": run_id,
        "execution_date": str(execution_date) if execution_date else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "git_commit": git_commit(root),
        "python_version": platform.python_version(),
        "copy_models": copy_models,
        "metrics": collect_metrics(root),
        "files": files,
        "models": models,
        "missing_models": missing_models,
        "manifest_path": str(run_dir / "manifest.json"),
    }

    with open(run_dir / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4, default=str)

    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Archive the artifacts of a BrainScan pipeline run."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execution-date", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="Record model metadata without copying the files.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    manifest = archive_run(
        run_id=args.run_id,
        execution_date=args.execution_date,
        output_root=args.output_root,
        copy_models=not args.no_models,
    )

    print(f"Artifacts archived to: {Path(manifest['manifest_path']).parent}")
    print(f"Files: {len(manifest['files'])} | Models: {len(manifest['models'])}")
    print(f"Manifest: {manifest['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
