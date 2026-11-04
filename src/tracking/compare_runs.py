"""Compare MLflow runs and identify the best configuration.

Examples
--------
List the experiments stored in the tracking backend:

    python -m src.tracking.compare_runs --list-experiments

Compare every classification run by test accuracy:

    python -m src.tracking.compare_runs \
        --experiment brainscan-classification \
        --metric test/accuracy

Compare detection runs by mAP@0.5:0.95 and tag the winner:

    python -m src.tracking.compare_runs \
        --experiment brainscan-detection \
        --metric test/mAP50-95 \
        --tag-best

Point at a remote server with ``MLFLOW_TRACKING_URI=http://localhost:5000``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.tracking.mlflow_utils import (
    ROOT_DIR,
    resolve_settings,
)

REPORT_DIR = ROOT_DIR / "reports" / "mlflow"

DEFAULT_METRICS = {
    "classification": ["test/accuracy", "test/f1_macro", "validation/accuracy"],
    "detection": ["test/mAP50-95", "test/mAP50", "test/precision", "test/recall"],
}

DEFAULT_PARAM_CANDIDATES = [
    "model.name",
    "model.image_size",
    "training.learning_rate",
    "training.epochs",
    "training.batch_size",
    "training.batch",
    "training.imgsz",
    "training.model",
    "finetuning.enabled",
    "finetuning.learning_rate",
    "finetuning.epochs",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare MLflow runs for the brain-tumor project."
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Experiment name (default: from config/env or 'brainscan').",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Metric used to rank runs (e.g. test/accuracy, test/mAP50-95).",
    )
    parser.add_argument(
        "--kind",
        choices=["classification", "detection"],
        default=None,
        help="Used to pick sensible default metrics.",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=None,
        help="Parameter to display (repeatable). Defaults to a curated list.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=50,
        help="Maximum number of runs to fetch (default: 50).",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Override the MLflow tracking URI.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Comparison CSV path (default: reports/mlflow/<experiment>.csv).",
    )
    parser.add_argument(
        "--tag-best",
        action="store_true",
        help="Tag the winning run with best_run=true.",
    )
    parser.add_argument(
        "--alias",
        default=None,
        help="Register this model alias on the best run's model version.",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List experiments and exit.",
    )
    return parser.parse_args()


def list_experiments() -> None:
    client = MlflowClient()
    experiments = client.search_experiments()
    if not experiments:
        print("No MLflow experiments found.")
        return
    print(f"{'id':>4}  {'name':<40} {'lifecycle':>10}  artifact_location")
    for experiment in experiments:
        print(
            f"{experiment.experiment_id:>4}  "
            f"{experiment.name:<40} "
            f"{experiment.lifecycle_stage:>10}  "
            f"{experiment.artifact_location}"
        )


def resolve_metric(metric: str | None, kind: str | None, columns) -> str:
    candidates = []
    if metric:
        candidates.append(metric)
    if kind:
        candidates.extend(DEFAULT_METRICS[kind])
    else:
        for names in DEFAULT_METRICS.values():
            candidates.extend(names)

    for candidate in candidates:
        if f"metrics.{candidate}" in columns:
            return candidate

    available = sorted(
        column[len("metrics.") :]
        for column in columns
        if column.startswith("metrics.")
    )
    raise SystemExit(
        "No ranking metric found. Available metrics:\n  "
        + "\n  ".join(available)
    )


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


def fetch_runs(experiment_name: str, max_runs: int):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise SystemExit(
            f"Experiment '{experiment_name}' not found. "
            "Use --list-experiments to see the available ones."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time ASC"],
        max_results=max_runs,
    )
    return experiment, runs


def select_params(columns, requested: list[str] | None, max_params: int = 7):
    available = [column for column in columns if column.startswith("params.")]
    available_names = [column[len("params.") :] for column in available]

    if requested:
        return [name for name in requested if name in available_names]

    curated = [
        name for name in DEFAULT_PARAM_CANDIDATES if name in available_names
    ]
    if curated:
        return curated[:max_params]
    return available_names[:max_params]


def format_table(records, params, metrics) -> str:
    header = ["rank", "run_name", *params, *metrics]
    rows = []
    for rank, record in enumerate(records, 1):
        row = [str(rank), record["run_name"] or record["run_id"][:8]]
        for param in params:
            value = record["params"].get(param, "")
            if value is None or (
                isinstance(value, float) and value != value
            ):
                value = ""
            row.append(str(value))
        for metric in metrics:
            value = record["metrics"].get(metric)
            row.append("" if value is None else f"{value:.4f}")
        rows.append(row)

    widths = [
        max(len(header[index]), *(len(row[index]) for row in rows))
        if rows
        else len(header[index])
        for index in range(len(header))
    ]

    def render(row):
        return "  ".join(
            cell.ljust(width) for cell, width in zip(row, widths)
        )

    lines = [render(header), render(["-" * width for width in widths])]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)


def main():
    args = parse_args()

    settings = resolve_settings()
    tracking_uri = args.tracking_uri or settings["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)

    print(f"MLflow tracking URI: {tracking_uri}")

    if args.list_experiments:
        list_experiments()
        return

    experiment_name = args.experiment or settings["experiment_name"]
    experiment, runs = fetch_runs(experiment_name, args.max_runs)

    if runs.empty:
        raise SystemExit(f"Experiment '{experiment_name}' has no runs yet.")

    metric = resolve_metric(args.metric, args.kind, runs.columns)
    params = select_params(runs.columns, args.param)

    metric_columns = [
        candidate
        for candidate in (
            args.metric,
            *(DEFAULT_METRICS[args.kind] if args.kind else []),
            metric,
        )
        if candidate and f"metrics.{candidate}" in runs.columns
    ]
    # Preserve order while removing duplicates.
    metric_columns = list(dict.fromkeys(metric_columns))

    records = []
    for _, run in runs.iterrows():
        record_params = {
            column[len("params.") :]: run[column]
            for column in runs.columns
            if column.startswith("params.")
        }
        record_metrics = {
            column[len("metrics.") :]: clean_value(run[column])
            for column in runs.columns
            if column.startswith("metrics.")
        }
        run_name = run.get("tags.mlflow.runName", "")
        if not isinstance(run_name, str):
            run_name = ""
        records.append(
            {
                "run_id": run["run_id"],
                "run_name": run_name,
                "status": run.get("status", ""),
                "start_time": str(run.get("start_time", "")),
                "params": record_params,
                "metrics": record_metrics,
            }
        )

    ranked = sorted(
        records,
        key=lambda record: (
            record["metrics"].get(metric) is not None,
            record["metrics"].get(metric) or float("-inf"),
        ),
        reverse=True,
    )

    print(f"\nExperiment: {experiment.name} (id={experiment.experiment_id})")
    print(f"Runs: {len(ranked)} | ranked by: {metric}\n")
    print(format_table(ranked, params, metric_columns))

    best = ranked[0]
    best_value = best["metrics"].get(metric)
    print(
        f"\nBest run: {best['run_name'] or best['run_id'][:8]} "
        f"({metric}={best_value:.4f})"
        if best_value is not None
        else f"\nBest run: {best['run_name'] or best['run_id'][:8]}"
    )

    output = args.output or (REPORT_DIR / f"{experiment.name}_comparison.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output, index=False)

    summary = {
        "experiment": experiment.name,
        "experiment_id": experiment.experiment_id,
        "tracking_uri": tracking_uri,
        "ranking_metric": metric,
        "best_run": {
            "run_id": best["run_id"],
            "run_name": best["run_name"],
            "metric": metric,
            "value": best_value,
        },
        "runs": ranked,
    }
    summary_path = output.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, default=str)

    print(f"Comparison saved to: {output}")
    print(f"Summary saved to:    {summary_path}")

    client = MlflowClient()

    if args.tag_best:
        client.set_tag(best["run_id"], "best_run", "true")
        client.set_tag(best["run_id"], "best_metric", metric)
        client.set_tag(best["run_id"], "best_metric_value", str(best_value))
        print(f"Tagged best run: {best['run_id']}")

    if args.alias:
        versions = client.search_model_versions(f"run_id='{best['run_id']}'")
        if not versions:
            print(
                "No registered model version found for the best run; "
                "the model was logged as an artifact only."
            )
        else:
            latest = max(versions, key=lambda version: int(version.version))
            client.set_registered_model_alias(
                latest.name, args.alias, latest.version
            )
            print(
                f"Alias '{args.alias}' -> {latest.name} "
                f"version {latest.version}"
            )


if __name__ == "__main__":
    main()
