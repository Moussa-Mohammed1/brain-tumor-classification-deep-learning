from __future__ import annotations

import io

import pytest

pytest.importorskip("fastapi", reason="fastapi is not installed")

from PIL import Image  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api import main as api_main  # noqa: E402
from api.inference import ModelUnavailableError  # noqa: E402
from api.main import create_app  # noqa: E402


CLASSIFICATION_PAYLOAD = {
    "predicted_class": "Glioma",
    "confidence": 0.91,
    "probabilities": {
        "Glioma": 0.91,
        "Meningioma": 0.04,
        "No Tumor": 0.03,
        "Pituitary": 0.02,
    },
}

DETECTIONS = [
    {
        "class_id": 0,
        "class_name": "Glioma",
        "confidence": 0.88,
        "box": {"x1": 10, "y1": 20, "x2": 110, "y2": 120},
    },
    {
        "class_id": 3,
        "class_name": "Pituitary",
        "confidence": 0.72,
        "box": {"x1": 30, "y1": 40, "x2": 70, "y2": 90},
    },
]


class FakeClassifier:
    model_name = "fake-classification"
    available = True

    def __init__(self, payload=None, error=None):
        self.payload = payload or CLASSIFICATION_PAYLOAD
        self.error = error

    def predict(self, image):
        if self.error is not None:
            raise self.error
        return self.payload


class FakeDetector:
    model_name = "fake-detection"
    available = True

    def __init__(self, detections=None, error=None):
        self.detections = detections if detections is not None else DETECTIONS
        self.error = error

    def predict(self, image):
        if self.error is not None:
            raise self.error
        return self.detections


def image_bytes(size=(64, 64), image_format="PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "black").save(buffer, format=image_format)
    return buffer.getvalue()


def make_client(classifier=None, detector=None) -> TestClient:
    app = create_app(
        classification_predictor=classifier or FakeClassifier(),
        detection_predictor=detector or FakeDetector(),
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# Meta endpoints
# ---------------------------------------------------------------------------
def test_root_endpoint():
    response = make_client().get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "BrainScan API"
    assert "/classify" in payload["endpoints"]


def test_health_reports_available_models():
    response = make_client().get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["models"] == {
        "classification": True,
        "detection": True,
    }


def test_health_is_degraded_when_a_model_is_missing():
    classifier = FakeClassifier()
    classifier.available = False

    payload = make_client(classifier=classifier).get("/health").json()

    assert payload["status"] == "degraded"
    assert payload["models"]["classification"] is False


def test_default_app_does_not_load_models_at_import():
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert set(response.json()["models"]) == {"classification", "detection"}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def test_classify_returns_prediction():
    response = make_client().post(
        "/classify",
        files={"file": ("scan.png", image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "scan.png"
    assert payload["predicted_class"] == "Glioma"
    assert payload["confidence"] == pytest.approx(0.91)
    assert payload["probabilities"]["Glioma"] == pytest.approx(0.91)
    assert payload["model"] == "fake-classification"
    assert payload["latency_ms"] >= 0


def test_classify_requires_a_file():
    response = make_client().post("/classify")

    assert response.status_code == 422


def test_classify_rejects_invalid_image():
    response = make_client().post(
        "/classify",
        files={"file": ("scan.png", b"not-an-image", "image/png")},
    )

    assert response.status_code == 400
    assert "valid image" in response.json()["detail"]


def test_classify_rejects_too_large_file(monkeypatch):
    monkeypatch.setattr(api_main, "MAX_IMAGE_BYTES", 10)

    response = make_client().post(
        "/classify",
        files={"file": ("scan.png", b"x" * 100, "image/png")},
    )

    assert response.status_code == 413


def test_classify_rejects_unsupported_extension():
    response = make_client().post(
        "/classify",
        files={"file": ("scan.txt", image_bytes(), "text/plain")},
    )

    assert response.status_code == 400


def test_classify_returns_503_when_model_is_missing():
    classifier = FakeClassifier(error=ModelUnavailableError("missing model"))

    response = make_client(classifier=classifier).post(
        "/classify",
        files={"file": ("scan.png", image_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert "missing model" in response.json()["detail"]


def test_classify_hides_internal_errors():
    classifier = FakeClassifier(error=RuntimeError("secret stack trace"))

    response = make_client(classifier=classifier).post(
        "/classify",
        files={"file": ("scan.png", image_bytes(), "image/png")},
    )

    assert response.status_code == 500
    assert "secret stack trace" not in response.json()["detail"]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def test_detect_returns_boxes():
    response = make_client().post(
        "/detect",
        files={"file": ("scan.png", image_bytes(size=(128, 96)), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["filename"] == "scan.png"
    assert payload["image_width"] == 128
    assert payload["image_height"] == 96
    assert payload["detections"][0]["class_name"] == "Glioma"
    assert payload["detections"][0]["box"] == {
        "x1": 10.0,
        "y1": 20.0,
        "x2": 110.0,
        "y2": 120.0,
    }


def test_detect_accepts_an_image_without_tumour():
    response = make_client(detector=FakeDetector(detections=[])).post(
        "/detect",
        files={"file": ("scan.png", image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 0
    assert payload["detections"] == []


def test_detect_returns_503_when_model_is_missing():
    detector = FakeDetector(error=ModelUnavailableError("missing weights"))

    response = make_client(detector=detector).post(
        "/detect",
        files={"file": ("scan.png", image_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert "missing weights" in response.json()["detail"]


def test_detect_rejects_tiny_image():
    response = make_client().post(
        "/detect",
        files={"file": ("scan.png", image_bytes(size=(8, 8)), "image/png")},
    )

    assert response.status_code == 400
    assert "too small" in response.json()["detail"]
