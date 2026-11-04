from __future__ import annotations

import pytest
import requests
from PIL import Image

from streamlit_app import api_client
from streamlit_app.api_client import (
    APIError,
    ClassificationResult,
    Detection,
    DetectionResult,
    check_health,
    classify_image,
    detect_tumors,
    draw_detections,
    normalize_base_url,
)


class FakeResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_normalize_base_url():
    assert normalize_base_url("localhost:8000/") == "http://localhost:8000"
    assert normalize_base_url(" http://api.local/ ") == "http://api.local"
    assert normalize_base_url("https://example.com") == "https://example.com"
    assert normalize_base_url("").startswith("http")


# ---------------------------------------------------------------------------
# Classification payloads
# ---------------------------------------------------------------------------
def test_classification_result_from_dict_payload():
    result = ClassificationResult.from_payload(
        {
            "predicted_class": "Glioma",
            "confidence": 0.93,
            "probabilities": {
                "Glioma": 0.93,
                "Meningioma": 0.03,
                "No Tumor": 0.02,
                "Pituitary": 0.02,
            },
            "model": "brainscan-classification",
            "latency_ms": 42.1,
        }
    )

    assert result.predicted_class == "Glioma"
    assert result.confidence == pytest.approx(0.93)
    assert result.probabilities["Meningioma"] == pytest.approx(0.03)
    assert result.model == "brainscan-classification"
    assert result.latency_ms == pytest.approx(42.1)


def test_classification_result_derives_prediction_from_probabilities():
    result = ClassificationResult.from_payload(
        {"probabilities": {"Glioma": 0.1, "Meningioma": 0.8}}
    )

    assert result.predicted_class == "Meningioma"
    assert result.confidence == pytest.approx(0.8)


def test_classification_result_accepts_score_list():
    result = ClassificationResult.from_payload(
        {
            "scores": [
                {"label": "Glioma", "score": 0.2},
                {"label": "Pituitary", "score": 0.75},
            ]
        }
    )

    assert result.probabilities == {
        "Glioma": pytest.approx(0.2),
        "Pituitary": pytest.approx(0.75),
    }
    assert result.predicted_class == "Pituitary"


def test_classification_result_clamps_confidence():
    result = ClassificationResult.from_payload(
        {"predicted_class": "Glioma", "confidence": 1.4}
    )

    assert result.confidence == 1.0


def test_classification_result_rejects_non_mapping():
    with pytest.raises(APIError):
        ClassificationResult.from_payload(["not", "a", "mapping"])


# ---------------------------------------------------------------------------
# Detection payloads
# ---------------------------------------------------------------------------
def test_detection_result_parses_boxes_and_class_ids():
    result = DetectionResult.from_payload(
        {
            "detections": [
                {
                    "class_id": 0,
                    "confidence": 0.9,
                    "box": {"x1": 1, "y1": 2, "x2": 10, "y2": 20},
                },
                {
                    "class": "Pituitary",
                    "confidence": 0.5,
                    "bbox": [30, 40, 50, 60],
                    "normalized": True,
                },
            ],
            "model": "brainscan-detection",
            "latency_ms": 55.0,
        }
    )

    assert len(result.detections) == 2
    assert result.detections[0].class_name == "Glioma"
    assert result.detections[0].class_id == 0
    assert result.detections[1].class_name == "Pituitary"
    assert result.detections[1].scaled_box(100, 200) == (3000, 8000, 5000, 12000)
    assert result.model == "brainscan-detection"


def test_detection_result_skips_invalid_boxes():
    result = DetectionResult.from_payload(
        {
            "detections": [
                {"class_name": "Glioma", "box": [10, 10, 5, 5]},
                {"class_name": "Glioma"},
                "not-a-mapping",
                {"class_name": "Glioma", "box": [0, 0, 10, 10]},
            ]
        }
    )

    assert len(result.detections) == 1
    assert result.detections[0].class_name == "Glioma"


def test_detection_result_accepts_string_box():
    result = DetectionResult.from_payload(
        {"detections": [{"class_name": "Glioma", "box": "1, 2, 30, 40"}]}
    )

    detection = result.detections[0]
    assert (detection.x1, detection.y1, detection.x2, detection.y2) == (
        1.0,
        2.0,
        30.0,
        40.0,
    )


def test_detection_result_empty_payload():
    result = DetectionResult.from_payload({})

    assert result.detections == ()
    assert result.to_dict()["count"] == 0


# ---------------------------------------------------------------------------
# HTTP calls
# ---------------------------------------------------------------------------
def test_classify_image_posts_multipart(monkeypatch):
    captured = {}

    def fake_post(url, files=None, timeout=None):
        captured["url"] = url
        captured["files"] = files
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "predicted_class": "Glioma",
                "confidence": 0.9,
                "probabilities": {"Glioma": 0.9},
            }
        )

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    result = classify_image(
        "localhost:8000/", b"fake-image", "scan.png", timeout=10
    )

    assert captured["url"] == "http://localhost:8000/classify"
    assert captured["files"]["file"][0] == "scan.png"
    assert captured["files"]["file"][1] == b"fake-image"
    assert captured["files"]["file"][2] == "image/png"
    assert captured["timeout"] == 10
    assert result.predicted_class == "Glioma"


def test_detect_tumors_posts_multipart(monkeypatch):
    captured = {}

    def fake_post(url, files=None, timeout=None):
        captured["url"] = url
        captured["files"] = files
        return FakeResponse({"detections": []})

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    result = detect_tumors("http://api.local", b"fake-image", "scan.jpg")

    assert captured["url"] == "http://api.local/detect"
    assert captured["files"]["file"][2] == "image/jpeg"
    assert result.detections == ()


def test_check_health(monkeypatch):
    captured = {}

    def fake_get(url, timeout=None):
        captured["url"] = url
        captured["timeout"] = timeout
        return FakeResponse(
            {"status": "ok", "models": {"classification": True}}
        )

    monkeypatch.setattr(api_client.requests, "get", fake_get)

    health = check_health("http://api.local/")

    assert captured["url"] == "http://api.local/health"
    assert health["models"]["classification"] is True


def test_http_error_exposes_api_detail(monkeypatch):
    def fake_post(url, files=None, timeout=None):
        return FakeResponse(
            {"detail": [{"msg": "file must be an image"}]},
            status_code=422,
        )

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    with pytest.raises(APIError) as error:
        classify_image("http://api.local", b"data", "scan.jpg")

    assert error.value.status_code == 422
    assert "file must be an image" in str(error.value)


def test_connection_error_is_wrapped(monkeypatch):
    def fake_post(url, files=None, timeout=None):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    with pytest.raises(APIError) as error:
        detect_tumors("http://api.local", b"data", "scan.jpg")

    assert "Cannot reach the API" in str(error.value)
    assert "uvicorn api.main:app" in str(error.value)


def test_invalid_json_is_reported(monkeypatch):
    def fake_post(url, files=None, timeout=None):
        return FakeResponse(ValueError("not json"), text="<html>oops</html>")

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    with pytest.raises(APIError) as error:
        classify_image("http://api.local", b"data", "scan.jpg")

    assert "invalid JSON" in str(error.value)


def test_empty_image_is_rejected(monkeypatch):
    def fake_post(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("HTTP request should not be sent")

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    with pytest.raises(APIError):
        classify_image("http://api.local", b"", "scan.jpg")


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def test_draw_detections_changes_the_image():
    image = Image.new("RGB", (120, 80), "black")
    detection = Detection(
        class_name="Glioma",
        confidence=0.87,
        x1=10,
        y1=10,
        x2=80,
        y2=60,
    )

    annotated = draw_detections(image, [detection])

    assert annotated.size == image.size
    assert annotated.mode == "RGB"
    assert annotated.tobytes() != image.tobytes()
    assert image.tobytes() == Image.new("RGB", (120, 80), "black").tobytes()


def test_draw_detections_without_boxes_returns_a_copy():
    image = Image.new("RGB", (20, 20), "white")

    annotated = draw_detections(image, [])

    assert annotated.tobytes() == image.tobytes()
    assert annotated is not image
