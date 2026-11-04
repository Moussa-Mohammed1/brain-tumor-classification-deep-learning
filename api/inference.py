"""Model loading and preprocessing for the BrainScan FastAPI service.

TensorFlow, tf_keras, tensorflow_hub and Ultralytics are imported lazily so
the API can start (and the test suite can run) without the heavy ML stack.
The first prediction loads the corresponding model and keeps it in memory.

Model resolution order
----------------------
Classification:
    1. ``BRAINSCAN_CLASSIFICATION_MODEL`` environment variable,
    2. ``models/classification/best_model.keras``,
    3. ``models/classification/best_model_phase1.keras``,
    4. ``models/classification/best_model_phase2.keras``.

Detection:
    1. ``BRAINSCAN_DETECTION_MODEL`` environment variable,
    2. ``models/detection/best.pt``,
    3. ``models/detection/yolov8n_brain_tumor/weights/best.pt``.

Detection thresholds can be overridden with ``BRAINSCAN_DETECTION_CONF``,
``BRAINSCAN_DETECTION_IOU`` and ``BRAINSCAN_DETECTION_IMGSZ``.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger("brainscan.api.inference")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASSIFICATION_MODEL_CANDIDATES = (
    "models/classification/best_model.keras",
    "models/classification/best_model_phase1.keras",
    "models/classification/best_model_phase2.keras",
)
DETECTION_MODEL_CANDIDATES = (
    "models/detection/best.pt",
    "models/detection/yolov8n_brain_tumor/weights/best.pt",
)
DETECTION_CLASSES_PATH = PROJECT_ROOT / "models" / "detection" / "classes.json"

CLASS_NAMES = ("Glioma", "Meningioma", "No Tumor", "Pituitary")
IMAGE_SIZE = (299, 299)

MODEL_NAME_CLASSIFICATION = "brainscan-classification"
MODEL_NAME_DETECTION = "brainscan-detection"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MAX_IMAGE_BYTES = int(os.environ.get("BRAINSCAN_MAX_IMAGE_MB", "15")) * 1024 * 1024
MAX_IMAGE_PIXELS = int(
    os.environ.get("BRAINSCAN_MAX_IMAGE_PIXELS", "50000000")
)

_DETECTION_BACKEND_WARMED = False


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class ModelUnavailableError(RuntimeError):
    """The model file or one of its ML dependencies is missing."""


class InferenceError(RuntimeError):
    """The model could not produce a valid prediction."""


class InvalidImageError(ValueError):
    """The uploaded bytes are not a usable image."""


class ImageTooLargeError(ValueError):
    """The uploaded image exceeds the configured size limit."""


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def _first_existing(
    project_root: Path,
    candidates: tuple[str, ...],
) -> Path:
    for relative in candidates:
        path = project_root / relative
        if path.is_file():
            return path
    return project_root / candidates[0]


def resolve_classification_model_path(
    project_root: str | Path | None = None,
) -> Path:
    override = os.environ.get("BRAINSCAN_CLASSIFICATION_MODEL")
    if override:
        return Path(override).expanduser()
    root = Path(project_root or PROJECT_ROOT)
    return _first_existing(root, CLASSIFICATION_MODEL_CANDIDATES)


def resolve_detection_model_path(
    project_root: str | Path | None = None,
) -> Path:
    override = os.environ.get("BRAINSCAN_DETECTION_MODEL")
    if override:
        return Path(override).expanduser()
    root = Path(project_root or PROJECT_ROOT)
    return _first_existing(root, DETECTION_MODEL_CANDIDATES)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_image(
    data: bytes,
    filename: str | None = None,
    content_type: str | None = None,
) -> Image.Image:
    """Validate an upload and return an RGB ``PIL.Image``.

    Raises:
        ImageTooLargeError: the file or its pixel count is too large.
        InvalidImageError: the file is empty, not an image or unreadable.
    """
    if not data:
        raise InvalidImageError("The uploaded file is empty.")

    if len(data) > MAX_IMAGE_BYTES:
        raise ImageTooLargeError(
            f"The image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit."
        )

    suffix = Path(filename or "").suffix.lower()
    if suffix and suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise InvalidImageError(
            f"Unsupported extension '{suffix}'. Allowed: {allowed}."
        )

    if content_type and not content_type.startswith("image/"):
        raise InvalidImageError(
            f"Unsupported content type '{content_type}'. Expected an image."
        )

    try:
        with Image.open(io.BytesIO(data)) as opened:
            width, height = opened.size
            if width < 16 or height < 16:
                raise InvalidImageError(
                    "The image is too small (minimum 16 × 16 pixels)."
                )
            if width * height > MAX_IMAGE_PIXELS:
                raise ImageTooLargeError(
                    f"The image has too many pixels ({width} × {height})."
                )
            opened.load()
            return opened.convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        if isinstance(exc, (InvalidImageError, ImageTooLargeError)):
            raise
        raise InvalidImageError(
            "The uploaded file is not a valid image."
        ) from exc


def preprocess_classification(
    image: Image.Image,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Resize and scale an image exactly like the training pipeline.

    ``src/classification/dataset.py`` resizes to 299×299 with bilinear
    interpolation and applies ``Rescaling(1 / 255)``.
    """
    resized = image.convert("RGB").resize(
        image_size, resample=Image.Resampling.BILINEAR
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------
def warm_up_detection_backend() -> None:
    """Load the PyTorch/Ultralytics native extensions once, before TensorFlow.

    Some TensorFlow and PyTorch builds both register LLVM command-line options.
    If Ultralytics is imported and used *after* TensorFlow, PyTorch can abort
    the process with ``LLVM ERROR: inconsistency in registered CommandLine
    options``. Running one tiny YOLO inference first loads the native libraries
    in a safe order. The helper is a no-op when Ultralytics or the detection
    weights are absent, and can be disabled with
    ``BRAINSCAN_DISABLE_BACKEND_WARMUP=1``.
    """
    global _DETECTION_BACKEND_WARMED
    if _DETECTION_BACKEND_WARMED:
        return
    _DETECTION_BACKEND_WARMED = True

    if os.environ.get(
        "BRAINSCAN_DISABLE_BACKEND_WARMUP", ""
    ).lower() in {"1", "true", "yes"}:
        return

    if not resolve_detection_model_path().is_file():
        return

    try:
        from ultralytics import YOLO

        model = YOLO(str(resolve_detection_model_path()))
        model.predict(
            source=Image.new("RGB", (64, 64), "black"),
            imgsz=64,
            conf=0.99,
            verbose=False,
        )
        logger.info(
            "Detection backend warmed up before TensorFlow import."
        )
    except ImportError:
        return
    except Exception:  # noqa: BLE001 - optional warm-up must not block startup
        logger.warning("Detection backend warm-up failed", exc_info=True)


class ClassificationPredictor:
    """Lazy InceptionV3 classifier.

    ``predict`` returns a dict matching the fields of
    ``api.schemas.ClassificationResponse`` (minus filename/model/latency).
    """

    model_name = MODEL_NAME_CLASSIFICATION

    def __init__(
        self,
        model_path: str | Path | None = None,
        class_names: tuple[str, ...] | list[str] = CLASS_NAMES,
        image_size: tuple[int, int] = IMAGE_SIZE,
    ) -> None:
        self.model_path = Path(
            model_path or resolve_classification_model_path()
        )
        self.class_names = tuple(class_names)
        self.image_size = image_size
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.model_path.is_file()

    def load(self):
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            if not self.available:
                raise ModelUnavailableError(
                    f"Classification model not found: {self.model_path}"
                )

            # Load the PyTorch/Ultralytics native libraries first: doing so
            # after TensorFlow can abort the process on some builds (see
            # ``warm_up_detection_backend``).
            warm_up_detection_backend()

            try:
                import tensorflow_hub as hub
                import tf_keras as keras
            except ImportError as exc:  # pragma: no cover - depends on env
                raise ModelUnavailableError(
                    "tensorflow, tf-keras and tensorflow-hub are required "
                    "for classification."
                ) from exc

            logger.info("Loading classification model from %s", self.model_path)
            self._model = keras.models.load_model(
                str(self.model_path),
                custom_objects={"KerasLayer": hub.KerasLayer},
                compile=False,
            )
            return self._model

    def predict(self, image: Image.Image) -> dict:
        model = self.load()
        batch = preprocess_classification(image, self.image_size)

        try:
            raw = model.predict(batch, verbose=0)
        except Exception as exc:  # noqa: BLE001 - surfaced as an API error
            raise InferenceError(
                f"Classification inference failed: {exc}"
            ) from exc

        probabilities = np.asarray(raw, dtype=np.float64).reshape(-1)
        if probabilities.size != len(self.class_names):
            raise InferenceError(
                "The model returned "
                f"{probabilities.size} scores for "
                f"{len(self.class_names)} classes."
            )

        probabilities = np.clip(probabilities, 0.0, 1.0)
        index = int(np.argmax(probabilities))
        probability_map = {
            name: float(value)
            for name, value in zip(self.class_names, probabilities)
        }

        return {
            "predicted_class": self.class_names[index],
            "confidence": float(probabilities[index]),
            "probabilities": probability_map,
        }


class DetectionPredictor:
    """Lazy YOLOv8 detector.

    ``predict`` returns a list of dicts matching
    ``api.schemas.DetectionItem``.
    """

    model_name = MODEL_NAME_DETECTION

    def __init__(
        self,
        model_path: str | Path | None = None,
        classes_path: str | Path | None = None,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(
            model_path or resolve_detection_model_path()
        )
        self.classes_path = Path(classes_path or DETECTION_CLASSES_PATH)
        self.conf = (
            conf
            if conf is not None
            else float(os.environ.get("BRAINSCAN_DETECTION_CONF", "0.25"))
        )
        self.iou = (
            iou
            if iou is not None
            else float(os.environ.get("BRAINSCAN_DETECTION_IOU", "0.6"))
        )
        self.imgsz = (
            imgsz
            if imgsz is not None
            else int(os.environ.get("BRAINSCAN_DETECTION_IMGSZ", "640"))
        )
        self.device = device or os.environ.get("BRAINSCAN_DEVICE") or None
        self._model: Any = None
        self._class_names: dict[int, str] | None = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.model_path.is_file()

    def load(self):
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            if not self.available:
                raise ModelUnavailableError(
                    f"Detection model not found: {self.model_path}"
                )

            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover - depends on env
                raise ModelUnavailableError(
                    "ultralytics and torch are required for detection."
                ) from exc

            logger.info("Loading detection model from %s", self.model_path)
            self._model = YOLO(str(self.model_path))
            self._class_names = self._resolve_class_names(self._model)
            return self._model

    def _resolve_class_names(self, model: Any) -> dict[int, str]:
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, (list, tuple)):
            return {index: str(value) for index, value in enumerate(names)}

        if self.classes_path.is_file():
            try:
                with open(self.classes_path, "r", encoding="utf-8") as file:
                    loaded = json.load(file)
                if isinstance(loaded, list):
                    return {
                        index: str(value)
                        for index, value in enumerate(loaded)
                    }
                if isinstance(loaded, dict):
                    return {
                        int(key): str(value)
                        for key, value in loaded.items()
                    }
            except (OSError, ValueError, TypeError):
                logger.warning(
                    "Could not read class names from %s", self.classes_path
                )

        return {index: name for index, name in enumerate(CLASS_NAMES)}

    def predict(self, image: Image.Image) -> list[dict]:
        model = self.load()

        kwargs: dict[str, Any] = {
            "source": image,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device

        try:
            results = model.predict(**kwargs)
        except Exception as exc:  # noqa: BLE001 - surfaced as an API error
            raise InferenceError(f"Detection inference failed: {exc}") from exc

        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []

        coordinates = _to_numpy(getattr(boxes, "xyxy", None))
        confidences = _to_numpy(getattr(boxes, "conf", None)).reshape(-1)
        classes = _to_numpy(getattr(boxes, "cls", None)).reshape(-1)

        if coordinates.size == 0:
            return []
        coordinates = coordinates.reshape(-1, 4)
        length = min(len(coordinates), len(confidences), len(classes))

        names = self._class_names or {
            index: name for index, name in enumerate(CLASS_NAMES)
        }
        detections = []
        for index in range(length):
            x1, y1, x2, y2 = (float(value) for value in coordinates[index])
            confidence = _clamp(float(confidences[index]))
            class_id = int(classes[index])
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": names.get(class_id, f"class_{class_id}"),
                    "confidence": confidence,
                    "box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },
                }
            )

        detections.sort(
            key=lambda detection: detection["confidence"], reverse=True
        )
        return detections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,))
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
