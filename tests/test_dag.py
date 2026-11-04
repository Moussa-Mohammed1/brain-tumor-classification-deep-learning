"""Validate that the Airflow DAG parses and keeps its expected structure.

The test is skipped automatically when Apache Airflow is not installed in the
current environment, so the rest of the suite can run without the dedicated
orchestration virtualenv.
"""

import os
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("airflow", reason="apache-airflow is not installed")

# Parsing a DAG does not need a metadata database, but Airflow resolves its
# home directory on import, so point it at a throwaway location.
os.environ.setdefault(
    "AIRFLOW_HOME", str(Path(tempfile.gettempdir()) / "brainscan-airflow-home")
)

from airflow.models import DagBag  # noqa: E402  (import after importorskip)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAGS_FOLDER = PROJECT_ROOT / "airflow" / "dags"
DAG_ID = "brainscan_pipeline"

EXPECTED_TASKS = {
    "prepare.check_environment",
    "prepare.prepare_detection_dataset",
    "prepare.validate_datasets",
    "classification.train_classification",
    "classification.evaluate_classification",
    "classification.compare_classification_runs",
    "detection.train_detection",
    "detection.evaluate_detection",
    "detection.compare_detection_runs",
    "finalize.archive_artifacts",
    "finalize.summarize_pipeline",
}

EXPECTED_DEPENDENCIES = {
    "prepare.check_environment": {"prepare.prepare_detection_dataset"},
    "prepare.prepare_detection_dataset": {"prepare.validate_datasets"},
    "prepare.validate_datasets": {
        "classification.train_classification",
        "detection.train_detection",
    },
    "classification.train_classification": {
        "classification.evaluate_classification"
    },
    "classification.evaluate_classification": {
        "classification.compare_classification_runs"
    },
    "classification.compare_classification_runs": {
        "finalize.archive_artifacts"
    },
    "detection.train_detection": {"detection.evaluate_detection"},
    "detection.evaluate_detection": {"detection.compare_detection_runs"},
    "detection.compare_detection_runs": {"finalize.archive_artifacts"},
    "finalize.archive_artifacts": {"finalize.summarize_pipeline"},
    "finalize.summarize_pipeline": set(),
}


@pytest.fixture(scope="module")
def dag():
    bag = DagBag(dag_folder=str(DAGS_FOLDER), include_examples=False)

    assert bag.import_errors == {}, (
        f"The DAG failed to import: {bag.import_errors}"
    )
    assert DAG_ID in bag.dags, f"{DAG_ID} not found in {sorted(bag.dags)}"

    return bag.dags[DAG_ID]


def test_dag_parses_without_import_errors(dag):
    assert dag.dag_id == DAG_ID


def test_dag_exposes_all_pipeline_tasks(dag):
    assert {task.task_id for task in dag.tasks} == EXPECTED_TASKS


def test_task_dependencies_match_the_expected_graph(dag):
    actual = {
        task.task_id: {downstream.task_id for downstream in task.downstream_list}
        for task in dag.tasks
    }

    assert actual == EXPECTED_DEPENDENCIES


def test_dag_is_manual_and_not_catching_up(dag):
    assert dag.schedule_interval is None
    assert dag.catchup is False
    assert dag.max_active_runs == 1


def test_default_args_enable_retries_and_status_logging(dag):
    assert dag.default_args["retries"] == 1
    assert dag.default_args["retry_delay"].total_seconds() == 300
    assert callable(dag.default_args["on_success_callback"])
    assert callable(dag.default_args["on_failure_callback"])
    assert callable(dag.default_args["on_retry_callback"])


def test_dag_parameters_are_documented(dag):
    params = dag.params

    assert params["force_prepare"] is False
    assert params["verify_dependencies"] is True
    assert params["archive_models"] is True
