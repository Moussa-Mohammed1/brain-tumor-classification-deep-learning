"""Pre-flight checks executed before the Airflow pipeline starts.

The checks are intentionally dependency-light (standard library only) so they
can run inside the Airflow environment even when TensorFlow/Ultralytics are
installed in a separate project virtualenv. The optional dependency probe
launches that project interpreter as a subprocess.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

REQUIRED_CONFIGS = (
    "configs/classification.yaml",
    "configs/detection.yaml",
)

REQUIRED_MODULES = (
    "src/classification/train.py",
    "src/classification/evaluate.py",
    "src/detection/train.py",
    "src/detection/evaluate.py",
    "src/data/prepare_detection.py",
    "src/orchestration/validate_data.py",
)

DEPENDENCY_IMPORTS = (
    "tensorflow",
    "tf_keras",
    "tensorflow_hub",
    "ultralytics",
    "mlflow",
    "yaml",
    "cv2",
)

MIN_FREE_DISK_BYTES = 2 * 1024**3  # 2 GiB


def count_images(directory: Path) -> int:
    """Recursively count image files under ``directory``."""
    if not directory.is_dir():
        return 0
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _resolve_executable(python_executable: str | None) -> str:
    executable = python_executable or sys.executable
    if Path(executable).is_file():
        return executable
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    raise FileNotFoundError(f"Python executable not found: {executable}")


def check_environment(
    python_executable: str | None = None,
    verify_dependencies: bool = True,
    project_root: str | Path | None = None,
    dependency_timeout: int = 600,
) -> dict:
    """Validate the project layout, data availability and runtime.

    Returns a JSON-friendly summary. Raises ``RuntimeError`` with the complete
    list of problems when a blocking issue is found, so the Airflow task fails
    early with an actionable message.
    """
    root = Path(project_root or PROJECT_ROOT).resolve()
    errors: list[str] = []
    warnings: list[str] = []

    # -- static project files ---------------------------------------------
    missing_configs = [
        path for path in REQUIRED_CONFIGS if not (root / path).is_file()
    ]
    missing_modules = [
        path for path in REQUIRED_MODULES if not (root / path).is_file()
    ]
    if missing_configs:
        errors.append(f"Missing config files: {missing_configs}")
    if missing_modules:
        errors.append(f"Missing pipeline modules: {missing_modules}")

    # -- classification data (required) -----------------------------------
    classification_raw = root / "data" / "classification"
    classification_split = root / "data" / "classification_split"

    if not classification_raw.is_dir():
        errors.append(f"Missing raw classification data: {classification_raw}")

    split_counts = {}
    if not classification_split.is_dir():
        errors.append(f"Missing classification splits: {classification_split}")
    else:
        for split in ("Train", "Validation", "Test"):
            split_dir = classification_split / split
            count = count_images(split_dir)
            split_counts[split] = count
            if not split_dir.is_dir():
                errors.append(f"Missing classification split: {split_dir}")
            elif count == 0:
                errors.append(f"No images found in {split_dir}")

    # -- detection data (prepared by the first DAG task) ------------------
    detection_yaml = root / "data" / "detection" / "data.yaml"
    if detection_yaml.is_file():
        detection_status = "present"
    else:
        detection_status = "missing (will be prepared)"
        warnings.append(
            "data/detection/data.yaml is missing; the prepare task will build it."
        )

    # -- runtime ----------------------------------------------------------
    python_info = {}
    try:
        python = _resolve_executable(python_executable)
        python_info["executable"] = python
        result = subprocess.run(
            [python, "-c", "import platform; print(platform.python_version())"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        python_info["version"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"Could not run the project interpreter: {exc}")
        python = None

    if verify_dependencies and python and not errors:
        code = "import " + ", ".join(DEPENDENCY_IMPORTS)
        try:
            result = subprocess.run(
                [python, "-c", code],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=dependency_timeout,
            )
            if result.returncode != 0:
                tail = (result.stderr or result.stdout).strip().splitlines()
                errors.append(
                    "Project dependencies are not importable with "
                    f"{python}: {tail[-1] if tail else 'unknown error'}"
                )
            else:
                python_info["dependencies"] = "ok"
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"Dependency check failed: {exc}")

    # -- disk space --------------------------------------------------------
    disk = shutil.disk_usage(root)
    if disk.free < MIN_FREE_DISK_BYTES:
        warnings.append(
            f"Low disk space: {disk.free / 1024**3:.1f} GiB free"
        )

    summary = {
        "project_root": str(root),
        "platform": platform.platform(),
        "python": python_info,
        "classification": {
            "raw_dir": str(classification_raw),
            "split_dir": str(classification_split),
            "images": split_counts,
        },
        "detection": {
            "data_yaml": str(detection_yaml),
            "status": detection_status,
        },
        "disk_free_gb": round(disk.free / 1024**3, 2),
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }

    if errors:
        raise RuntimeError(
            "Pre-flight checks failed:\n- " + "\n- ".join(errors)
        )

    print(json.dumps(summary, indent=4))
    return summary
