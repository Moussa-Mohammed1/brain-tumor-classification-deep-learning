from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from api import inference
from api.inference import (
    ClassificationPredictor,
    DetectionPredictor,
    ImageTooLargeError,
    InferenceError,
    InvalidImageError,
    load_image,
    preprocess_classification,
    resolve_classification_model_path,
    resolve_detection_model_path,
)


def image_bytes(size=(64, 64), image_format="PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "black").save(buffer, format=image_format)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------
def test_load_image_returns_rgb():
    image = load_image(image_bytes(size=(80, 60)), "scan.png", "image/png")

    assert image.mode == "RGB"
    assert image.size == (80, 60)


def test_load_image_rejects_empty_data():
    with pytest.raises(InvalidImageError):
        load_image(b"", "scan.png", "image/png")


def test_load_image_rejects_bad_extension():
    with pytest.raises(InvalidImageError) as error:
        load_image(image_bytes(), "scan.txt", "text/plain")

    assert "Unsupported extension" in str(error.value)


def test_load_image_rejects_bad_content_type():
    with pytest.raises(InvalidImageError) as error:
        load_image(image_bytes(), "scan.png", "application/pdf")

    assert "Unsupported content type" in str(error.value)


def test_load_image_rejects_corrupted_bytes():
    with pytest.raises(InvalidImageError):
        load_image(b"not an image", "scan.png", "image/png")


def test_load_image_rejects_tiny_image():
    with pytest.raises(InvalidImageError) as error:
        load_image(image_bytes(size=(8, 8)), "scan.png", "image/png")

    assert "too small" in str(error.value)


def test_load_image_rejects_large_file(monkeypatch):
    monkeypatch.setattr(inference, "MAX_IMAGE_BYTES", 10)

    with pytest.raises(ImageTooLargeError):
        load_image(b"x" * 100, "scan.png", "image/png")


def test_load_image_rejects_large_pixel_count(monkeypatch):
    monkeypatch.setattr(inference, "MAX_IMAGE_PIXELS", 100)

    with pytest.raises(ImageTooLargeError):
        load_image(image_bytes(size=(20, 20)), "scan.png", "image/png")


# ---------------------------------------------------------------------------
# Classification preprocessing
# ---------------------------------------------------------------------------
def test_preprocess_classification_matches_training():
    image = Image.new("RGB", (100, 50), (255, 0, 0))

    batch = preprocess_classification(image)

    assert batch.shape == (1, 299, 299, 3)
    assert batch.dtype == np.float32
    assert batch.min() == pytest.approx(0.0)
    assert batch.max() == pytest.approx(1.0)


class FakeKerasModel:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float64)
        self.batch_shape = None

    def predict(self, batch, verbose=0):
        self.batch_shape = batch.shape
        return self.scores.reshape(1, -1)


def test_classification_predictor_maps_scores_to_classes():
    predictor = ClassificationPredictor(
        model_path="/does/not/exist.keras",
        class_names=("Glioma", "Meningioma", "No Tumor", "Pituitary"),
    )
    predictor._model = FakeKerasModel([0.1, 0.2, 0.6, 0.1])

    result = predictor.predict(Image.new("RGB", (40, 40), "black"))

    assert result["predicted_class"] == "No Tumor"
    assert result["confidence"] == pytest.approx(0.6)
    assert result["probabilities"]["Glioma"] == pytest.approx(0.1)
    assert predictor._model.batch_shape == (1, 299, 299, 3)


def test_classification_predictor_rejects_wrong_score_count():
    predictor = ClassificationPredictor(model_path="/does/not/exist.keras")
    predictor._model = FakeKerasModel([0.5, 0.5])

    with pytest.raises(InferenceError):
        predictor.predict(Image.new("RGB", (40, 40), "black"))


def test_classification_predictor_requires_a_model_file():
    predictor = ClassificationPredictor(model_path="/does/not/exist.keras")

    assert predictor.available is False
    with pytest.raises(inference.ModelUnavailableError):
        predictor.load()


# ---------------------------------------------------------------------------
# Detection preprocessing / prediction
# ---------------------------------------------------------------------------
class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)
        self.cls = FakeTensor(cls)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLO:
    names = {0: "Glioma", 1: "Meningioma", 3: "Pituitary"}

    def __init__(self, results):
        self.results = results
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return self.results


def test_detection_predictor_builds_sorted_boxes():
    boxes = FakeBoxes(
        xyxy=[[5, 6, 20, 30], [1, 2, 30, 40]],
        conf=[0.4, 0.9],
        cls=[1, 0],
    )
    model = FakeYOLO([FakeResult(boxes)])
    predictor = DetectionPredictor(
        model_path="/does/not/exist.pt",
        conf=0.3,
        iou=0.5,
        imgsz=320,
    )
    predictor._model = model
    predictor._class_names = predictor._resolve_class_names(model)

    detections = predictor.predict(Image.new("RGB", (80, 60), "black"))

    assert len(detections) == 2
    assert detections[0]["class_name"] == "Glioma"
    assert detections[0]["confidence"] == pytest.approx(0.9)
    assert detections[1]["class_name"] == "Meningioma"
    assert model.kwargs["imgsz"] == 320
    assert model.kwargs["conf"] == 0.3
    assert model.kwargs["iou"] == 0.5


def test_detection_predictor_handles_empty_results():
    predictor = DetectionPredictor(model_path="/does/not/exist.pt")
    predictor._model = FakeYOLO([])

    assert predictor.predict(Image.new("RGB", (80, 60), "black")) == []


def test_detection_predictor_handles_empty_boxes():
    predictor = DetectionPredictor(model_path="/does/not/exist.pt")
    predictor._model = FakeYOLO([FakeResult(None)])

    assert predictor.predict(Image.new("RGB", (80, 60), "black")) == []


def test_class_names_fall_back_to_classes_json(tmp_path: Path):
    classes_path = tmp_path / "classes.json"
    classes_path.write_text(
        json.dumps(["Glioma", "Meningioma", "No Tumor", "Pituitary"]),
        encoding="utf-8",
    )
    predictor = DetectionPredictor(
        model_path="/does/not/exist.pt",
        classes_path=classes_path,
    )

    class ModelWithoutNames:
        names = None

    names = predictor._resolve_class_names(ModelWithoutNames())

    assert names[2] == "No Tumor"


def test_detection_predictor_requires_a_model_file():
    predictor = DetectionPredictor(model_path="/does/not/exist.pt")

    assert predictor.available is False
    with pytest.raises(inference.ModelUnavailableError):
        predictor.load()


# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------
def test_classification_path_prefers_best_model(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("BRAINSCAN_CLASSIFICATION_MODEL", raising=False)
    best = tmp_path / "models" / "classification" / "best_model.keras"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"model")

    assert resolve_classification_model_path(tmp_path) == best


def test_classification_path_falls_back_to_phase2(tmp_path: Path):
    phase2 = (
        tmp_path
        / "models"
        / "classification"
        / "best_model_phase2.keras"
    )
    phase2.parent.mkdir(parents=True)
    phase2.write_bytes(b"model")

    assert resolve_classification_model_path(tmp_path) == phase2


def test_classification_path_honours_environment(monkeypatch, tmp_path: Path):
    override = tmp_path / "custom.keras"
    monkeypatch.setenv("BRAINSCAN_CLASSIFICATION_MODEL", str(override))

    assert resolve_classification_model_path(tmp_path) == override


def test_detection_path_honours_environment(monkeypatch, tmp_path: Path):
    override = tmp_path / "custom.pt"
    monkeypatch.setenv("BRAINSCAN_DETECTION_MODEL", str(override))

    assert resolve_detection_model_path(tmp_path) == override


def test_detection_path_falls_back_to_training_weights(tmp_path: Path):
    weights = (
        tmp_path
        / "models"
        / "detection"
        / "yolov8n_brain_tumor"
        / "weights"
        / "best.pt"
    )
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")

    assert resolve_detection_model_path(tmp_path) == weights
