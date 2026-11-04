from pathlib import Path

import pytest

from src.tracking import mlflow_utils
from src.tracking.mlflow_utils import (
    MAX_PARAM_LENGTH,
    MLflowTracker,
    _param_value,
    _sanitize_key,
    flatten_mapping,
    resolve_settings,
)


def test_flatten_mapping_handles_nested_blocks():
    flat = flatten_mapping(
        {
            "training": {"learning_rate": 0.001, "epochs": 20},
            "model": {"image_size": [299, 299]},
        }
    )

    assert flat["training.learning_rate"] == 0.001
    assert flat["training.epochs"] == 20
    assert flat["model.image_size"] == [299, 299]


def test_custom_separator_is_used():
    flat = flatten_mapping({"a": {"b": 1}}, separator="/")

    assert flat == {"a/b": 1}


def test_param_value_serialises_special_types():
    assert _param_value(True) == "true"
    assert _param_value(False) == "false"
    assert _param_value(None) == "null"
    assert _param_value([1, 2]) == "[1, 2]"


def test_param_value_is_truncated():
    value = _param_value("x" * (MAX_PARAM_LENGTH + 50))

    assert len(value) == MAX_PARAM_LENGTH


def test_sanitize_key_removes_unsupported_characters():
    assert _sanitize_key("metrics/precision(B)") == "metrics/precision_B_"
    assert _sanitize_key("") == "unnamed"


def test_settings_defaults_to_local_store():
    settings = resolve_settings({})

    assert settings["tracking_uri"].startswith("file:")
    assert settings["experiment_name"]


def test_environment_overrides_config(monkeypatch):
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "from-env")

    settings = resolve_settings({"experiment_name": "from-config"})

    assert settings["experiment_name"] == "from-env"


def test_disabled_tracker_is_a_noop():
    with MLflowTracker({"enabled": False}) as tracker:
        tracker.log_params({"training": {"epochs": 1}})
        tracker.log_metrics({"accuracy": 1.0})
        tracker.log_artifacts("does/not/exist")
        tracker.set_tags({"stage": "test"})

    assert tracker.run_id is None


requires_mlflow = pytest.mark.skipif(
    mlflow_utils.mlflow is None, reason="mlflow is not installed"
)


@requires_mlflow
def test_tracker_logs_params_metrics_and_artifacts(tmp_path: Path):
    tracking_uri = tmp_path / "mlruns"

    with MLflowTracker(
        {
            "tracking_uri": str(tracking_uri),
            "experiment_name": "pytest-experiment",
        },
        run_name="pytest-run",
        tags={"stage": "test"},
    ) as tracker:
        assert tracker.run_id is not None
        tracker.log_params(
            {"training": {"learning_rate": 0.001, "epochs": 2}}
        )
        tracker.log_metrics({"accuracy": 0.9, "loss": 0.1}, prefix="test/")
        tracker.log_dict({"model": "inception_v3"}, "model_card.json", "model")
        run_id = tracker.run_id

    client = mlflow_utils.MlflowClient()
    run = client.get_run(run_id)

    assert run.data.params["training.learning_rate"] == "0.001"
    assert run.data.metrics["test/accuracy"] == pytest.approx(0.9)
    assert run.data.tags["stage"] == "test"

    artifacts = client.list_artifacts(run_id, "model")
    assert any(item.path.endswith("model_card.json") for item in artifacts)
