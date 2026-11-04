# brain-tumor-classification-deep-learning

## Experiment tracking (MLflow)

Every training and evaluation run is tracked with MLflow: hyperparameters,
per-epoch metrics, final metrics, plots, configuration files and the retained
model (registered in the MLflow Model Registry).

### Where runs are stored

By default runs go to the local `mlruns/` directory (no server needed). To use
a remote server, export the tracking URI before running any script:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

The YAML `mlflow` blocks in `configs/classification.yaml` and
`configs/detection.yaml` control the experiment name and the registered model
name. Environment variables take precedence over the YAML values.

### Run the pipelines

```bash
# Classification: two-phase InceptionV3 training + evaluation
python -m src.classification.train
python -m src.classification.evaluate

# Detection: YOLOv8 training + evaluation
python -m src.detection.train
python -m src.detection.evaluate
```

| Pipeline | Experiment | Run name | Registered model |
|---|---|---|---|
| Classification train | `brainscan-classification` | `classification-<timestamp>` | `brainscan-classification` |
| Classification eval | `brainscan-classification` | `classification-eval-<timestamp>` | — |
| Detection train | `brainscan-detection` | `detection-<timestamp>` | `brainscan-detection` |
| Detection eval | `brainscan-detection` | `detection-eval-<timestamp>` | — |

Logged metrics follow `<split>/<metric>` naming, e.g. `test/accuracy`,
`test/f1_macro`, `test/mAP50-95`, plus per-class values such as
`test/per_class/Glioma/recall`.

### Open the UI

```bash
mlflow ui --backend-store-uri mlruns
```

### Compare experiments

`src/tracking/compare_runs.py` ranks runs, saves a CSV/JSON report under
`reports/mlflow/`, and can tag the winner and promote its model version:

```bash
# Classification runs, ranked by test accuracy
python -m src.tracking.compare_runs \
    --experiment brainscan-classification \
    --metric test/accuracy --tag-best

# Detection runs, ranked by mAP@0.5:0.95, best version exposed as alias "production"
python -m src.tracking.compare_runs \
    --experiment brainscan-detection \
    --kind detection \
    --metric test/mAP50-95 \
    --tag-best --alias production
```

List the available experiments:

```bash
python -m src.tracking.compare_runs --list-experiments
```

### Use the registered models

```python
import mlflow.pyfunc

# InceptionV3 classifier: returns class, confidence and per-class probabilities
classifier = mlflow.pyfunc.load_model("models:/brainscan-classification/best")
classifier.predict(["scan.jpg"])

# YOLOv8 detector: returns bounding boxes and class names
detector = mlflow.pyfunc.load_model("models:/brainscan-detection/production")
detector.predict(["scan.jpg"])
```

Tracking is failure-tolerant: if MLflow is not installed or the server is
unreachable, training continues and only prints a warning.

## Pipeline orchestration (Airflow)

The end-to-end workflow is defined in `airflow/dags/brainscan_pipeline.py` and
chains dataset preparation, data validation, training, evaluation, MLflow
tracking and artifact archiving.

### Task graph

```
check_environment
  └─ prepare_detection_dataset
       └─ validate_datasets
            ├─ train_classification ─ evaluate_classification ─ compare_classification_runs ─┐
            └─ train_detection      ─ evaluate_detection      ─ compare_detection_runs      ─┤
                                                                                             └─ archive_artifacts
                                                                                                  └─ summarize_pipeline
```

Each task maps to an existing module, so the DAG orchestrates the project
instead of duplicating its logic:

| Task | Entry point |
|---|---|
| `prepare.check_environment` | `src.orchestration.checks` |
| `prepare.prepare_detection_dataset` | `src.data.prepare_detection` |
| `prepare.validate_datasets` | `src.orchestration.validate_data` |
| `classification.train_classification` | `src.classification.train` |
| `classification.evaluate_classification` | `src.classification.evaluate` |
| `classification.compare_classification_runs` | `src.tracking.compare_runs` |
| `detection.train_detection` | `src.detection.train` |
| `detection.evaluate_detection` | `src.detection.evaluate` |
| `detection.compare_detection_runs` | `src.tracking.compare_runs` |
| `finalize.archive_artifacts` | `src.orchestration.artifacts` |
| `finalize.summarize_pipeline` | inline |

### Install

Airflow pins older Flask/SQLAlchemy/Connexion versions that conflict with
TensorFlow and Streamlit, so install it in a **dedicated** virtualenv or
container, always with the official constraints file:

```bash
python3.10 -m venv .venv-airflow
source .venv-airflow/bin/activate
pip install --upgrade pip
pip install -r requirements/orchestration.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.10.txt"
```

The training/evaluation tasks run with the project interpreter (the one that
has TensorFlow and Ultralytics installed), which Airflow launches as a
subprocess. Point it at the right environment with:

```bash
export BRAINSCAN_PYTHON="$PWD/.venv/bin/python"
export MLFLOW_TRACKING_URI="$PWD/mlruns"   # optional, defaults to ./mlruns
```

### Run

```bash
export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/airflow/dags"
airflow db migrate          # first run only
airflow standalone          # UI + scheduler on http://localhost:8080
```

The DAG is manual (`schedule=None`, `catchup=False`, `max_active_runs=1`).
Trigger it from the UI or the CLI:

```bash
# Reuse the prepared datasets and the current models
airflow dags trigger brainscan_pipeline

# Rebuild data/detection and copy model files into the artifact archive
airflow dags trigger brainscan_pipeline \
    --conf '{"force_prepare": true, "archive_models": true}'
```

Supported `dag_run.conf` keys:

| Key | Default | Effect |
|---|---|---|
| `force_prepare` | `false` | Rebuild `data/detection` even if `data.yaml` exists |
| `verify_dependencies` | `true` | Import TensorFlow/Ultralytics before training |
| `archive_models` | `true` | Copy model weights into the artifact archive |

### Data validation

`src/orchestration/validate_data.py` checks the classification splits and the
YOLO dataset (class folders, matching image/label pairs, YOLO box format,
corrupted images) and writes `reports/orchestration/data_validation.json`.
The DAG task fails when a blocking error is found. Run it standalone with:

```bash
python -m src.orchestration.validate_data
```

### Artifacts and status

The final tasks write:

- `artifacts/airflow_runs/<run_id>/` — configs, `data.yaml`, reports and
  models, with a `manifest.json` containing SHA-256 hashes, the git commit,
  the Python version and a metrics snapshot.
- `reports/orchestration/run_summary.json` — validation status and final
  metrics for the run.
- `reports/orchestration/task_events.jsonl` — one JSON line per task status
  change (success, failure or retry), with task id, state, try number and log
  URL.

Every task has a 2-hour execution timeout by default, one retry with a 5-minute
delay, and per-task timeouts where training needs more (6 h classification,
12 h detection). The status callbacks append to the event log, so pipeline
health can be audited without opening the Airflow UI.

### Tests

```bash
# Orchestration helpers (no Airflow needed)
pytest tests/test_orchestration.py

# DAG structure (skipped when Airflow is not installed)
pytest tests/test_dag.py
```

## REST API (FastAPI)

`api/main.py` exposes both models over HTTP for the Streamlit UI or any other
client. Models are loaded lazily on the first prediction, so the service starts
instantly and `/health` still reports whether the files are available.

### Endpoints

| Method | Endpoint | Request | Response |
|---|---|---|---|
| `GET` | `/health` | — | `status`, `models`, `version` |
| `POST` | `/classify` | multipart field `file` | `predicted_class`, `confidence`, `probabilities`, `model`, `latency_ms` |
| `POST` | `/detect` | multipart field `file` | `detections[]`, `count`, `model`, `latency_ms`, `image_width`, `image_height` |

Classification classes follow the training order: `Glioma`, `Meningioma`,
`No Tumor`, `Pituitary` (alphabetical folder order). Detection boxes are pixel
coordinates in the original image (`xyxy`).

### Install and run

```bash
pip install -r requirements/api.txt
uvicorn api.main:app --reload --port 8000
```

- Interactive docs: <http://localhost:8000/docs>
- OpenAPI schema: <http://localhost:8000/openapi.json>
- Health check: <http://localhost:8000/health>

Example calls:

```bash
curl -F "file=@scan.jpg" http://localhost:8000/classify
curl -F "file=@scan.jpg" http://localhost:8000/detect
```

### Model resolution

The service looks for the models in this order:

| Model | Candidate paths |
|---|---|
| Classification | `BRAINSCAN_CLASSIFICATION_MODEL` → `models/classification/best_model.keras` → `best_model_phase2.keras` → `best_model_phase1.keras` |
| Detection | `BRAINSCAN_DETECTION_MODEL` → `models/detection/best.pt` → `models/detection/yolov8n_brain_tumor/weights/best.pt` |

If no file is found, `/health` reports `"degraded"` and the corresponding
endpoint returns HTTP 503 with an actionable message instead of a traceback.

### Configuration

| Variable | Default | Effect |
|---|---|---|
| `BRAINSCAN_CLASSIFICATION_MODEL` | resolved path | Override the classifier file |
| `BRAINSCAN_DETECTION_MODEL` | resolved path | Override the detector weights |
| `BRAINSCAN_DETECTION_CONF` | `0.25` | YOLOv8 confidence threshold |
| `BRAINSCAN_DETECTION_IOU` | `0.6` | YOLOv8 NMS IoU threshold |
| `BRAINSCAN_DETECTION_IMGSZ` | `640` | YOLOv8 inference size |
| `BRAINSCAN_DEVICE` | auto | `cpu`, `cuda`, `0`, … |
| `BRAINSCAN_MAX_IMAGE_MB` | `15` | Upload size limit |
| `BRAINSCAN_MAX_IMAGE_PIXELS` | `50000000` | Decompression-bomb guard |
| `BRAINSCAN_CORS_ORIGINS` | `*` | Comma-separated allowed origins |

### Error handling

| Status | Meaning |
|---|---|
| `400` | Empty, corrupted, too small or unsupported image |
| `413` | File or pixel count above the limit |
| `422` | Missing multipart `file` field |
| `500` | Inference failure (details are logged, not returned) |
| `503` | Model file or ML dependency missing |

### Files

| File | Role |
|---|---|
| `api/main.py` | FastAPI app, routes, upload validation, error mapping |
| `api/inference.py` | Image loading, preprocessing, lazy model loading, predictors |
| `api/schemas.py` | Pydantic request/response models |
| `tests/test_api.py` | Endpoint tests with injected fake predictors |
| `tests/test_api_inference.py` | Preprocessing, payload and model-path tests |

### Tests

```bash
pytest tests/test_api.py tests/test_api_inference.py
```

The tests inject fake predictors, so they run without TensorFlow, Ultralytics
or trained model files. The real estimators are only imported on first use.

## Web interface (Streamlit)

`streamlit_app/app.py` provides the interactive front end. It accepts an MRI
image (JPG, PNG or BMP, up to 15 MB) and offers two independent tasks:

- **Classification** — predicted class among Glioma / Meningioma / No Tumor /
  Pituitary, global confidence, a bar per class and a downloadable JSON result.
- **Detection & localisation** — YOLOv8 bounding boxes drawn on the original
  image, a table of class/confidence/coordinates and a downloadable JSON result.

The UI does **not** import TensorFlow or Ultralytics: it sends the image over
HTTP to the FastAPI service and renders the JSON response. Boxes are drawn
client-side with Pillow from the returned pixel coordinates.

### Install and run

```bash
pip install -r requirements/streamlit.txt

# Terminal 1 — FastAPI service (see the API contract below)
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Streamlit interface
streamlit run streamlit_app/app.py
```

Open <http://localhost:8501>. The API URL can be changed in the sidebar and is
tested with the **Test connection** button. Override the default with:

```bash
export BRAINSCAN_API_URL=http://127.0.0.1:8000
export BRAINSCAN_API_TIMEOUT=120      # inference timeout, seconds
```

### API contract expected by the UI

| Method | Endpoint | Request | Response |
|---|---|---|---|
| `GET` | `/health` | — | `{"status": "ok", "models": {"classification": true, "detection": true}}` |
| `POST` | `/classify` | multipart field `file` | `predicted_class`, `confidence`, `probabilities`, `model`, `latency_ms` |
| `POST` | `/detect` | multipart field `file` | `detections[{class_id, class_name, confidence, box{x1,y1,x2,y2}}]`, `model`, `latency_ms` |

Detection boxes are pixel coordinates in the original image (`xyxy`). A
detection may set `"normalized": true` to return coordinates in `[0, 1]`
instead; the client scales them before drawing.

Example classification response:

```json
{
  "predicted_class": "Glioma",
  "confidence": 0.9312,
  "probabilities": {
    "Glioma": 0.9312,
    "Meningioma": 0.0311,
    "No Tumor": 0.0204,
    "Pituitary": 0.0173
  },
  "model": "brainscan-classification",
  "latency_ms": 42.1
}
```

Example detection response:

```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "Glioma",
      "confidence": 0.88,
      "box": {"x1": 84, "y1": 62, "x2": 271, "y2": 240}
    }
  ],
  "model": "brainscan-detection",
  "latency_ms": 55.3
}
```

### Files

| File | Role |
|---|---|
| `streamlit_app/app.py` | Layout, upload, tabs, session state, result rendering |
| `streamlit_app/api_client.py` | HTTP client, payload normalisation, box drawing (no Streamlit import) |
| `tests/test_streamlit_client.py` | HTTP and parsing tests with mocked responses |
| `tests/test_streamlit_app.py` | Headless UI smoke tests via `streamlit.testing.v1.AppTest` |

### Tests

```bash
# HTTP client, parsing and drawing (works without Streamlit)
pytest tests/test_streamlit_client.py

# Headless interface (skipped when Streamlit is not installed)
pytest tests/test_streamlit_app.py
```

The client is failure-tolerant: unreachable API, timeout, HTTP error and
invalid JSON all surface as a readable message in the UI instead of a traceback.

