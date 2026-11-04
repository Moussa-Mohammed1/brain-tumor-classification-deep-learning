"""BrainScan end-to-end MLOps pipeline.

Task graph
----------
check_environment
  -> prepare_detection_dataset
  -> validate_datasets
       -> train_classification -> evaluate_classification -> compare_classification_runs
       -> train_detection      -> evaluate_detection      -> compare_detection_runs
                                                              -> archive_artifacts
                                                              -> summarize_pipeline

The DAG orchestrates the existing project entry points; it does not reimplement
training. Training/evaluation run with the project interpreter (``.venv`` by
default), while the lightweight Python callables run inside the Airflow
environment.

Configuration
-------------
Set ``BRAINSCAN_PYTHON`` to the interpreter that has TensorFlow/Ultralytics
installed and ``MLFLOW_TRACKING_URI`` to the tracking server. Both default to
the project virtualenv and the local ``mlruns/`` store.

Optional ``dag_run.conf`` keys:
    force_prepare        rebuild data/detection even if it already exists
    verify_dependencies  import TensorFlow/Ultralytics before training (true)
    archive_models       copy model files into artifacts/airflow_runs/<run_id>

This DAG is manual (``schedule=None``); change it to ``@weekly`` to run it on a
schedule.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger(__name__)

DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
PROJECT_PYTHON = os.environ.get(
    "BRAINSCAN_PYTHON",
    str(DEFAULT_PYTHON) if DEFAULT_PYTHON.is_file() else sys.executable,
)

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns")
)

BASH_ENV = {
    "PYTHONPATH": str(PROJECT_ROOT),
    "PYTHONUNBUFFERED": "1",
    "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
    "TF_CPP_MIN_LOG_LEVEL": os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2"),
}

EVENT_LOG = PROJECT_ROOT / "reports" / "orchestration" / "task_events.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _module_command(module: str, *arguments: str) -> str:
    """Build a shell command that runs a project module from the repo root."""
    pieces = [shlex.quote(PROJECT_PYTHON), "-m", module]
    pieces.extend(shlex.quote(str(argument)) for argument in arguments)
    return (
        f"cd {shlex.quote(str(PROJECT_ROOT))} && "
        f"exec {' '.join(pieces)}"
    )


def _conf(context: dict, key: str, default=None):
    """Read a key from ``dag_run.conf`` first, then from DAG params."""
    dag_run = context.get("dag_run")
    conf = (dag_run.conf if dag_run else None) or {}
    if key in conf:
        return conf[key]
    return (context.get("params") or {}).get(key, default)


def _append_event(event: str, context: dict) -> None:
    """Append a task status record to a JSONL log for later inspection."""
    task_instance = context.get("task_instance")
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "dag_id": context["dag"].dag_id if context.get("dag") else None,
        "task_id": getattr(task_instance, "task_id", None),
        "run_id": context.get("run_id"),
        "state": getattr(task_instance, "state", None),
        "try_number": getattr(task_instance, "try_number", None),
        "log_url": getattr(task_instance, "log_url", None),
        "exception": (
            str(context.get("exception"))
            if context.get("exception") is not None
            else None
        ),
    }

    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENT_LOG, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, default=str) + "\n")

    LOGGER.info("Airflow task event: %s", record)


def record_task_success(context: dict) -> None:
    _append_event("success", context)


def record_task_failure(context: dict) -> None:
    _append_event("failure", context)


def record_task_retry(context: dict) -> None:
    _append_event("retry", context)


# ---------------------------------------------------------------------------
# Python callables
# ---------------------------------------------------------------------------
def check_environment(**context) -> dict:
    """Fail fast when the project layout, data or runtime is incomplete."""
    from src.orchestration.checks import check_environment as run_checks

    return run_checks(
        python_executable=PROJECT_PYTHON,
        verify_dependencies=bool(
            _conf(context, "verify_dependencies", True)
        ),
        project_root=PROJECT_ROOT,
    )


def prepare_detection_dataset(**context) -> dict:
    """Build ``data/detection`` once; rebuild only when explicitly forced."""
    force = bool(_conf(context, "force_prepare", False))
    data_dir = PROJECT_ROOT / "data" / "detection"
    data_yaml = data_dir / "data.yaml"

    if data_yaml.is_file() and not force:
        counts = {
            split: (
                len(list((data_dir / split / "images").iterdir()))
                if (data_dir / split / "images").is_dir()
                else 0
            )
            for split in ("train", "valid", "test")
        }
        LOGGER.info(
            "Detection dataset already prepared (%s); skipping. "
            "Use conf {'force_prepare': true} to rebuild.",
            counts,
        )
        return {"status": "skipped", "images": counts}

    command = [PROJECT_PYTHON, "-m", "src.data.prepare_detection"]
    if force:
        command.append("--overwrite")

    LOGGER.info("Running: %s", " ".join(command))
    subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=True,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )

    return {"status": "rebuilt", "data_yaml": str(data_yaml)}


def archive_artifacts(**context) -> str:
    """Copy models/reports/configs into ``artifacts/airflow_runs/<run_id>``."""
    from src.orchestration.artifacts import archive_run

    dag_run = context["dag_run"]
    manifest = archive_run(
        run_id=dag_run.run_id,
        execution_date=dag_run.logical_date or dag_run.execution_date,
        copy_models=bool(_conf(context, "archive_models", True)),
        project_root=PROJECT_ROOT,
    )

    LOGGER.info(
        "Archived %d files and %d models to %s",
        len(manifest["files"]),
        len(manifest["models"]),
        manifest["manifest_path"],
    )
    return manifest["manifest_path"]


def summarize_pipeline(**context) -> str:
    """Write a compact JSON summary of validation, evaluation and artifacts."""
    from src.orchestration.artifacts import collect_metrics

    task_instance = context["ti"]

    validation_path = (
        PROJECT_ROOT
        / "reports"
        / "orchestration"
        / "data_validation.json"
    )
    validation = {}
    if validation_path.is_file():
        with open(validation_path, "r", encoding="utf-8") as file:
            validation = json.load(file)

    summary = {
        "run_id": context.get("run_id"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation": {
            "ok": validation.get("ok"),
            "errors": len(
                validation.get("classification", {}).get("errors", [])
            )
            + len(validation.get("detection", {}).get("errors", [])),
            "warnings": len(
                validation.get("classification", {}).get("warnings", [])
            )
            + len(validation.get("detection", {}).get("warnings", [])),
        },
        "metrics": collect_metrics(PROJECT_ROOT),
        "artifact_manifest": task_instance.xcom_pull(
            task_ids="finalize.archive_artifacts"
        ),
    }

    summary_path = (
        PROJECT_ROOT / "reports" / "orchestration" / "run_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, default=str)

    LOGGER.info("Pipeline summary:\n%s", json.dumps(summary, indent=2))

    # Keep the summary next to the archived artifacts too.
    manifest_path = summary.get("artifact_manifest")
    if manifest_path:
        shutil.copy2(summary_path, Path(manifest_path).parent / "run_summary.json")

    return str(summary_path)


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------
default_args = {
    "owner": "brainscan",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
    "on_success_callback": record_task_success,
    "on_failure_callback": record_task_failure,
    "on_retry_callback": record_task_retry,
}

with DAG(
    dag_id="brainscan_pipeline",
    description=(
        "Dataset preparation, classification/detection training, "
        "evaluation, MLflow tracking and artifact archiving."
    ),
    doc_md=__doc__,
    default_args=default_args,
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=24),
    tags=["brainscan", "mlops", "classification", "detection", "mlflow"],
    params={
        "force_prepare": False,
        "verify_dependencies": True,
        "archive_models": True,
    },
) as dag:

    # -- 1. Preparation & validation -------------------------------------
    with TaskGroup("prepare") as prepare_group:
        check_environment_task = PythonOperator(
            task_id="check_environment",
            python_callable=check_environment,
            execution_timeout=timedelta(minutes=30),
        )

        prepare_dataset_task = PythonOperator(
            task_id="prepare_detection_dataset",
            python_callable=prepare_detection_dataset,
            execution_timeout=timedelta(minutes=30),
        )

        validate_datasets_task = BashOperator(
            task_id="validate_datasets",
            bash_command=_module_command(
                "src.orchestration.validate_data",
                "--classification-dir",
                "data/classification_split",
                "--detection-yaml",
                "data/detection/data.yaml",
                "--output",
                "reports/orchestration/data_validation.json",
            ),
            env=BASH_ENV,
            append_env=True,
            execution_timeout=timedelta(minutes=30),
        )

        check_environment_task >> prepare_dataset_task >> validate_datasets_task

    # -- 2. Classification pipeline --------------------------------------
    with TaskGroup("classification") as classification_group:
        train_classification_task = BashOperator(
            task_id="train_classification",
            bash_command=_module_command("src.classification.train"),
            env=BASH_ENV,
            append_env=True,
            execution_timeout=timedelta(hours=6),
            retries=0,
        )

        evaluate_classification_task = BashOperator(
            task_id="evaluate_classification",
            bash_command=_module_command("src.classification.evaluate"),
            env=BASH_ENV,
            append_env=True,
            execution_timeout=timedelta(hours=2),
        )

        compare_classification_task = BashOperator(
            task_id="compare_classification_runs",
            bash_command=_module_command(
                "src.tracking.compare_runs",
                "--experiment",
                "brainscan-classification",
                "--kind",
                "classification",
                "--metric",
                "test/accuracy",
                "--tag-best",
                "--alias",
                "best",
            ),
            env=BASH_ENV,
            append_env=True,
            retries=2,
        )

        (
            train_classification_task
            >> evaluate_classification_task
            >> compare_classification_task
        )

    # -- 3. Detection pipeline -------------------------------------------
    with TaskGroup("detection") as detection_group:
        train_detection_task = BashOperator(
            task_id="train_detection",
            bash_command=_module_command("src.detection.train"),
            env=BASH_ENV,
            append_env=True,
            execution_timeout=timedelta(hours=12),
            retries=0,
        )

        evaluate_detection_task = BashOperator(
            task_id="evaluate_detection",
            bash_command=_module_command("src.detection.evaluate"),
            env=BASH_ENV,
            append_env=True,
            execution_timeout=timedelta(hours=2),
        )

        compare_detection_task = BashOperator(
            task_id="compare_detection_runs",
            bash_command=_module_command(
                "src.tracking.compare_runs",
                "--experiment",
                "brainscan-detection",
                "--kind",
                "detection",
                "--metric",
                "test/mAP50-95",
                "--tag-best",
                "--alias",
                "production",
            ),
            env=BASH_ENV,
            append_env=True,
            retries=2,
        )

        (
            train_detection_task
            >> evaluate_detection_task
            >> compare_detection_task
        )

    # -- 4. Archiving & summary ------------------------------------------
    with TaskGroup("finalize") as finalize_group:
        archive_task = PythonOperator(
            task_id="archive_artifacts",
            python_callable=archive_artifacts,
            execution_timeout=timedelta(minutes=30),
        )

        summarize_task = PythonOperator(
            task_id="summarize_pipeline",
            python_callable=summarize_pipeline,
            execution_timeout=timedelta(minutes=10),
        )

        archive_task >> summarize_task

    # -- Dependencies -----------------------------------------------------
    validate_datasets_task >> [
        train_classification_task,
        train_detection_task,
    ]

    compare_classification_task >> archive_task
    compare_detection_task >> archive_task
