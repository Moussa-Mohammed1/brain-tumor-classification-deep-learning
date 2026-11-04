"""MLflow tracking utilities shared by the classification and detection pipelines.

Both pipelines log their hyperparameters, metrics and artifacts to MLflow so
that runs can be compared and the retained model can be registered for reuse.

Design notes
------------
* Everything is defensive: if MLflow is not installed, disabled in the config,
  or the tracking server is unreachable, the helpers degrade to no-ops and
  training is never interrupted.
* By default runs are stored locally in ``mlruns/`` (no server required). Point
  ``MLFLOW_TRACKING_URI`` at a server (e.g. ``http://localhost:5000``) to use a
  remote store.
* Nested runs are supported: if a run is already active (for instance the
  Airflow DAG wraps a task in a run), a child run is created instead of failing.

Usage
-----
>>> from src.tracking.mlflow_utils import MLflowTracker
>>> with MLflowTracker(config.get("mlflow"), default_experiment="my-exp") as tracker:
...     tracker.log_params(config["training"])
...     tracker.log_metrics({"accuracy": 0.9})
...     tracker.log_artifacts("models/classification/best_model.keras", artifact_path="model")
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import tempfile
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping

try:  # MLflow is optional at runtime so data/EDA scripts stay lightweight.
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:  # pragma: no cover - exercised only in minimal environments
    mlflow = None
    MlflowClient = None


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_URI = "mlruns"
DEFAULT_EXPERIMENT = "brainscan"

# MLflow hard limits: 100 params per run, 250 chars per key, 500 per value.
MAX_PARAMS_PER_RUN = 100
MAX_KEY_LENGTH = 250
MAX_PARAM_LENGTH = 500

_INVALID_KEY_CHARS = re.compile(r"[^A-Za-z0-9_\-./ :]")

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings and small helpers
# ---------------------------------------------------------------------------
def utc_timestamp() -> str:
    """Return a sortable UTC timestamp suitable for run names."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def get_git_commit() -> str | None:
    """Return the short git commit of the working tree, if available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT_DIR,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def default_tags() -> dict:
    """Metadata attached to every run to make it identifiable later."""
    tags = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "source": "brain-tumor-classification-deep-learning",
    }
    commit = get_git_commit()
    if commit:
        tags["git_commit"] = commit
    return tags


def _normalize_tracking_uri(uri: str) -> str:
    """Anchor relative local paths to the project root.

    ``mlruns`` -> ``file:///.../mlruns`` while ``http://host:5000`` and
    ``file:/...`` are left untouched.
    """
    if "://" in uri or uri.startswith("file:"):
        return uri
    path = Path(uri)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.as_uri()


def resolve_settings(config: Mapping[str, Any] | None = None) -> dict:
    """Merge the ``mlflow`` config block with environment overrides.

    Priority: environment variable > YAML config > defaults.
    """
    config = dict(config or {})

    settings = {
        "enabled": bool(config.get("enabled", True)),
        "tracking_uri": (
            os.environ.get("MLFLOW_TRACKING_URI")
            or config.get("tracking_uri")
            or DEFAULT_TRACKING_URI
        ),
        "experiment_name": (
            os.environ.get("MLFLOW_EXPERIMENT_NAME")
            or config.get("experiment_name")
            or DEFAULT_EXPERIMENT
        ),
        "registered_model_name": config.get("registered_model_name"),
        "log_models": bool(config.get("log_models", True)),
    }
    settings["tracking_uri"] = _normalize_tracking_uri(
        str(settings["tracking_uri"])
    )

    if mlflow is None:
        if settings["enabled"]:
            LOGGER.warning(
                "MLflow is not installed: tracking is disabled. "
                "Install it with `pip install mlflow`."
            )
        settings["enabled"] = False

    return settings


def _sanitize_key(key: str) -> str:
    """Make a key acceptable to MLflow (allowed chars + max length)."""
    cleaned = _INVALID_KEY_CHARS.sub("_", str(key)).strip()
    return cleaned[:MAX_KEY_LENGTH] or "unnamed"


def _param_value(value: Any) -> str:
    """Convert a config value to a valid MLflow parameter string."""
    if isinstance(value, bool):
        value = "true" if value else "false"
    elif value is None:
        value = "null"
    elif isinstance(value, (list, tuple, set)):
        value = json.dumps(list(value))
    elif isinstance(value, Mapping):
        value = json.dumps(dict(value), sort_keys=True)
    return str(value)[:MAX_PARAM_LENGTH]


def flatten_mapping(
    mapping: Mapping[str, Any],
    parent: str = "",
    separator: str = ".",
) -> dict:
    """Flatten a nested config/mapping into a single-level dictionary.

    ``{"training": {"lr": 0.1}}`` becomes ``{"training.lr": 0.1}``.
    Lists are JSON-encoded so they survive as a single parameter.
    """
    flat: dict = {}
    for key, value in mapping.items():
        name = f"{parent}{separator}{key}" if parent else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_mapping(value, name, separator))
        else:
            flat[name] = value
    return flat


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and value == value  # filter NaN
    )


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
class MLflowTracker(AbstractContextManager):
    """Thin, failure-tolerant wrapper around an MLflow run.

    Use it as a context manager. On success the run is closed as ``FINISHED``;
    on exception it is closed as ``FAILED`` and the exception is re-raised.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        default_experiment: str = DEFAULT_EXPERIMENT,
        run_name: str | None = None,
        tags: Mapping[str, Any] | None = None,
        enabled: bool | None = None,
    ):
        self.settings = resolve_settings(config)

        # Only fall back to ``default_experiment`` when the caller did not
        # provide one explicitly (config or environment).
        if (
            not os.environ.get("MLFLOW_EXPERIMENT_NAME")
            and not (config or {}).get("experiment_name")
        ):
            self.settings["experiment_name"] = default_experiment

        if enabled is not None:
            self.settings["enabled"] = bool(enabled) and mlflow is not None

        self.enabled = bool(self.settings["enabled"])
        self.run_name = run_name
        self.tags = {**default_tags(), **dict(tags or {})}
        self.run_id: str | None = None
        self.experiment_id: str | None = None
        self._run = None
        self._param_count = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "MLflowTracker":
        if not self.enabled:
            LOGGER.info("MLflow tracking disabled; run is a no-op.")
            return self

        try:
            if mlflow.active_run() is None:
                mlflow.set_tracking_uri(self.settings["tracking_uri"])

            # The client captures the tracking URI at construction time, so it
            # must be created after ``set_tracking_uri``.
            client = MlflowClient()

            experiment_name = self.settings["experiment_name"]
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                self.experiment_id = client.create_experiment(experiment_name)
                LOGGER.info("Created MLflow experiment '%s'.", experiment_name)
            else:
                self.experiment_id = experiment.experiment_id

            nested = mlflow.active_run() is not None
            self._run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=self.run_name,
                tags=self.tags,
                nested=nested,
            )
            self.run_id = self._run.info.run_id
            LOGGER.info(
                "MLflow run started: %s (experiment=%s)",
                self.run_id,
                experiment_name,
            )
        except Exception as exc:  # pragma: no cover - depends on server state
            LOGGER.warning(
                "Could not start the MLflow run (%s); continuing without tracking.",
                exc,
            )
            self.enabled = False
        return self

    def __enter__(self) -> "MLflowTracker":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.end(status="FAILED" if exc_type else "FINISHED")
        return False  # never swallow the exception

    def end(self, status: str = "FINISHED") -> None:
        if not self.enabled or self._run is None or mlflow is None:
            return
        try:
            mlflow.end_run(status=status)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Could not close the MLflow run: %s", exc)
        finally:
            self._run = None

    # -- metadata ----------------------------------------------------------
    def set_tags(self, tags: Mapping[str, Any]) -> None:
        if not self._active():
            return
        try:
            mlflow.set_tags({str(k): str(v) for k, v in tags.items()})
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Could not log MLflow tags: %s", exc)

    def log_params(
        self,
        params: Mapping[str, Any],
        prefix: str = "",
        separator: str = ".",
    ) -> None:
        """Log a possibly nested mapping as MLflow parameters."""
        if not self._active():
            return

        flat = flatten_mapping(params, separator=separator)
        prepared = {}
        for key, value in flat.items():
            name = _sanitize_key(f"{prefix}{key}")
            if self._param_count + len(prepared) >= MAX_PARAMS_PER_RUN:
                LOGGER.warning(
                    "MLflow parameter limit (%d) reached; remaining parameters "
                    "were not logged. Use the config artifact for the full set.",
                    MAX_PARAMS_PER_RUN,
                )
                break
            prepared[name] = _param_value(value)

        if not prepared:
            return

        try:
            mlflow.log_params(prepared)
            self._param_count += len(prepared)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Could not log MLflow parameters: %s", exc)

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        prefix: str = "",
        step: int | None = None,
        separator: str = "/",
    ) -> None:
        """Log numeric metrics. Non-numeric values are skipped."""
        if not self._active():
            return

        flat = flatten_mapping(metrics, separator=separator)
        for key, value in flat.items():
            if not _is_number(value):
                LOGGER.debug("Skipping non-numeric metric '%s'.", key)
                continue
            name = _sanitize_key(f"{prefix}{key}")
            try:
                if step is None:
                    mlflow.log_metric(name, float(value))
                else:
                    mlflow.log_metric(name, float(value), step=int(step))
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Could not log metric '%s': %s", name, exc)

    def log_history(
        self,
        history: Mapping[str, Iterable[Any]] | Any,
        prefix: str = "",
        step_offset: int = 0,
    ) -> None:
        """Log per-epoch history (``keras.History`` or a dict of lists)."""
        if not self._active():
            return

        values = getattr(history, "history", history)
        if not isinstance(values, Mapping):
            return

        for key, series in values.items():
            for step, value in enumerate(series):
                if _is_number(value):
                    self.log_metrics(
                        {key: value},
                        prefix=prefix,
                        step=step_offset + step,
                    )

    # -- artifacts ---------------------------------------------------------
    def log_artifacts(
        self,
        paths: str | Path | Iterable[str | Path] | None,
        artifact_path: str | None = None,
    ) -> None:
        """Log files or directories. Missing paths are ignored with a warning."""
        if not self._active() or paths is None:
            return

        if isinstance(paths, (str, Path)):
            paths = [paths]

        for path in paths:
            local_path = Path(path)
            if not local_path.exists():
                LOGGER.warning("Artifact not found, skipping: %s", local_path)
                continue
            try:
                if local_path.is_dir():
                    mlflow.log_artifacts(
                        str(local_path), artifact_path=artifact_path
                    )
                else:
                    mlflow.log_artifact(
                        str(local_path), artifact_path=artifact_path
                    )
            except Exception as exc:  # pragma: no cover
                LOGGER.warning(
                    "Could not log artifact '%s': %s", local_path, exc
                )

    def log_dict(
        self,
        data: Mapping[str, Any],
        filename: str,
        artifact_path: str | None = None,
    ) -> None:
        """Write a dict as a JSON artifact."""
        if not self._active():
            return
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                target = Path(tmp_dir) / filename
                with open(target, "w", encoding="utf-8") as file:
                    json.dump(data, file, indent=4, default=str)
                self.log_artifacts(target, artifact_path=artifact_path)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Could not log JSON artifact '%s': %s", filename, exc)

    def log_text(
        self,
        text: str,
        filename: str,
        artifact_path: str | None = None,
    ) -> None:
        """Write a text/markdown artifact (e.g. a run summary)."""
        if not self._active():
            return
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                target = Path(tmp_dir) / filename
                target.write_text(text, encoding="utf-8")
                self.log_artifacts(target, artifact_path=artifact_path)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Could not log text artifact '%s': %s", filename, exc)

    # -- models ------------------------------------------------------------
    def log_keras_checkpoint(
        self,
        checkpoint_path: str | Path,
        class_names: list,
        image_size: tuple = (299, 299),
        registered_model_name: str | None = None,
        artifact_path: str = "model",
    ) -> bool:
        """Register a Keras checkpoint via the project's PyFunc wrapper.

        The wrapper reloads the ``.keras`` file with the TF Hub custom objects,
        so registration works even though MLflow's native TensorFlow flavor
        cannot serialise the Hub layer. Falls back to a plain artifact.
        """
        if not self._active():
            return False

        registered_model_name = (
            registered_model_name or self.settings.get("registered_model_name")
        )
        if not self.settings.get("log_models", True):
            LOGGER.info("Model registration disabled (log_models=false).")
            return False

        try:
            from src.tracking.keras_flavor import KerasPyfuncModel

            with tempfile.TemporaryDirectory() as tmp_dir:
                classes_path = Path(tmp_dir) / "classes.json"
                classes_path.write_text(
                    json.dumps(list(class_names)), encoding="utf-8"
                )
                mlflow.pyfunc.log_model(
                    artifact_path=artifact_path,
                    python_model=KerasPyfuncModel(
                        class_names=class_names,
                        image_size=image_size,
                    ),
                    artifacts={
                        "model": str(checkpoint_path),
                        "classes": str(classes_path),
                    },
                    pip_requirements=[
                        "tensorflow==2.21.0",
                        "tf-keras==2.21.0",
                        "tensorflow-hub==0.16.1",
                        "numpy==2.2.6",
                        "pillow==11.0.0",
                        "pandas==2.3.3",
                    ],
                    registered_model_name=registered_model_name,
                )
            self._tag_registered_version(registered_model_name)
            return True
        except Exception as exc:  # pragma: no cover - depends on MLflow/TF
            LOGGER.warning(
                "Could not register the Keras checkpoint (%s). "
                "The .keras file is still logged as an artifact.",
                exc,
            )
            self.log_artifacts(checkpoint_path, artifact_path=artifact_path)
            return False

    def log_yolo_model(
        self,
        weights_path: str | Path,
        classes_path: str | Path,
        registered_model_name: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> bool:
        """Log a YOLO checkpoint as an MLflow PyFunc model (best effort).

        Falls back to logging the raw weights as an artifact when the PyFunc
        flavor cannot be built.
        """
        if not self._active():
            return False

        registered_model_name = (
            registered_model_name or self.settings.get("registered_model_name")
        )
        if not self.settings.get("log_models", True):
            LOGGER.info("Model registration disabled (log_models=false).")
            return False

        try:
            from src.tracking.yolo_flavor import YoloPyfuncModel

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=YoloPyfuncModel(conf=conf, iou=iou),
                artifacts={
                    "weights": str(weights_path),
                    "classes": str(classes_path),
                },
                pip_requirements=[
                    "ultralytics==8.3.40",
                    "torch==2.5.1",
                    "torchvision==0.20.1",
                    "numpy==2.2.6",
                    "pillow==11.0.0",
                ],
                registered_model_name=registered_model_name,
            )
            self._tag_registered_version(registered_model_name)
            return True
        except Exception as exc:  # pragma: no cover - depends on MLflow/YOLO
            LOGGER.warning(
                "Could not log the YOLO PyFunc model (%s). "
                "Logging the weights as a plain artifact instead.",
                exc,
            )
            self.log_artifacts(weights_path, artifact_path="model")
            return False

    # -- internals ---------------------------------------------------------
    def _active(self) -> bool:
        return bool(self.enabled and self._run is not None and mlflow is not None)

    def _tag_registered_version(self, model_name: str | None) -> None:
        if not model_name or MlflowClient is None or not self.run_id:
            return
        try:
            versions = MlflowClient().search_model_versions(
                f"name='{model_name}'"
            )
            mine = [
                version
                for version in versions
                if version.run_id == self.run_id
            ]
            if mine:
                latest = max(mine, key=lambda version: int(version.version))
                self.set_tags(
                    {
                        "registered_model_name": model_name,
                        "registered_model_version": latest.version,
                    }
                )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Could not resolve the registered model version: %s", exc)


def start_tracker(
    config: Mapping[str, Any] | None = None,
    default_experiment: str = DEFAULT_EXPERIMENT,
    run_name: str | None = None,
    tags: Mapping[str, Any] | None = None,
) -> MLflowTracker:
    """Convenience factory returning a started tracker.

    Prefer the context-manager form in scripts::

        with MLflowTracker(...) as tracker:
            ...
    """
    return MLflowTracker(
        config,
        default_experiment=default_experiment,
        run_name=run_name,
        tags=tags,
    ).start()
