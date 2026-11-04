"""HTTP client used by the BrainScan Streamlit front end.

This module deliberately has no Streamlit import: it can be unit-tested
without a running Streamlit server and reused by scripts. It talks to the
FastAPI service over HTTP and converts JSON payloads into small immutable
dataclasses.

Expected API contract
----------------------
``GET /health``
    ``{"status": "ok", "models": {"classification": true, "detection": true}}``

``POST /classify`` (multipart field ``file``)
    ``{"filename": "scan.jpg", "predicted_class": "Glioma",
       "confidence": 0.93,
       "probabilities": {"Glioma": 0.93, "Meningioma": 0.03, ...},
       "model": "brainscan-classification", "latency_ms": 42.1}``

``POST /detect`` (multipart field ``file``)
    ``{"filename": "scan.jpg",
       "detections": [{"class_id": 0, "class_name": "Glioma",
                       "confidence": 0.88,
                       "box": {"x1": 10, "y1": 20, "x2": 110, "y2": 120}}],
       "model": "brainscan-detection", "latency_ms": 55.3}``

Boxes are pixel coordinates in the original image (``xyxy``). A detection
may set ``"normalized": true`` to provide values in ``[0, 1]`` instead.
"""

from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import requests
from PIL import Image, ImageDraw, ImageFont

DEFAULT_API_URL = os.environ.get(
    "BRAINSCAN_API_URL", "http://127.0.0.1:8000"
)
DEFAULT_TIMEOUT = float(os.environ.get("BRAINSCAN_API_TIMEOUT", "120"))
HEALTH_TIMEOUT = float(os.environ.get("BRAINSCAN_HEALTH_TIMEOUT", "5"))

CLASS_NAMES = ("Glioma", "Meningioma", "No Tumor", "Pituitary")

CLASS_COLORS = {
    "Glioma": "#e63946",
    "Meningioma": "#f4a261",
    "No Tumor": "#2a9d8f",
    "Pituitary": "#457b9d",
}
FALLBACK_COLOR = "#ff006e"


class APIError(RuntimeError):
    """Raised when the FastAPI service cannot be reached or answers badly."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def default_api_url() -> str:
    """Return the API base URL from the environment (or localhost)."""
    return DEFAULT_API_URL


def normalize_base_url(url: str | None) -> str:
    """Accept ``localhost:8000``, ``http://host/`` or an empty value."""
    value = (url or "").strip().rstrip("/")
    if not value:
        value = DEFAULT_API_URL
    if "://" not in value:
        value = f"http://{value}"
    return value


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassificationResult:
    predicted_class: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    model: str | None = None
    latency_ms: float | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> "ClassificationResult":
        if not isinstance(payload, dict):
            raise APIError("Unexpected classification response (expected JSON).")

        probabilities = _parse_probabilities(payload)

        predicted = (
            payload.get("predicted_class")
            or payload.get("class_name")
            or payload.get("label")
        )
        if predicted is None:
            raw_class = payload.get("class")
            if isinstance(raw_class, str) and not raw_class.isdigit():
                predicted = raw_class
        if predicted is None and probabilities:
            predicted = max(probabilities, key=probabilities.get)
        predicted = str(predicted or "Unknown")

        confidence = payload.get(
            "confidence", payload.get("score", payload.get("probability"))
        )
        confidence = _to_float(
            confidence, probabilities.get(predicted, 0.0)
        )

        return cls(
            predicted_class=predicted,
            confidence=_clamp(confidence),
            probabilities=probabilities,
            model=_optional_str(payload.get("model")),
            latency_ms=_optional_float(payload.get("latency_ms")),
        )

    def to_dict(self) -> dict:
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "model": self.model,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int | None = None
    normalized: bool = False

    def scaled_box(
        self,
        width: int,
        height: int,
    ) -> tuple[float, float, float, float]:
        """Return ``xyxy`` coordinates in pixels for a ``width``×``height`` image."""
        if not self.normalized:
            return self.x1, self.y1, self.x2, self.y2
        return (
            self.x1 * width,
            self.y1 * height,
            self.x2 * width,
            self.y2 * height,
        )

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "box": {
                "x1": self.x1,
                "y1": self.y1,
                "x2": self.x2,
                "y2": self.y2,
            },
            "normalized": self.normalized,
        }


@dataclass(frozen=True)
class DetectionResult:
    detections: tuple[Detection, ...] = ()
    model: str | None = None
    latency_ms: float | None = None
    image_width: int | None = None
    image_height: int | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> "DetectionResult":
        if not isinstance(payload, dict):
            raise APIError("Unexpected detection response (expected JSON).")

        raw = payload.get("detections")
        if raw is None:
            raw = payload.get("boxes", payload.get("predictions", []))
        if isinstance(raw, dict):
            raw = raw.get("detections", [])
        if not isinstance(raw, (list, tuple)):
            raise APIError("Detection response has an invalid 'detections' field.")

        detections = []
        for item in raw:
            detection = _parse_detection(item)
            if detection is not None:
                detections.append(detection)

        return cls(
            detections=tuple(detections),
            model=_optional_str(payload.get("model")),
            latency_ms=_optional_float(payload.get("latency_ms")),
            image_width=_optional_int(
                payload.get("image_width", payload.get("width"))
            ),
            image_height=_optional_int(
                payload.get("image_height", payload.get("height"))
            ),
        )

    def to_dict(self) -> dict:
        return {
            "detections": [
                detection.to_dict() for detection in self.detections
            ],
            "count": len(self.detections),
            "model": self.model,
            "latency_ms": self.latency_ms,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }


# ---------------------------------------------------------------------------
# HTTP calls
# ---------------------------------------------------------------------------
def check_health(
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict:
    """Call ``GET /health`` and return the decoded JSON object."""
    url = f"{normalize_base_url(base_url)}/health"
    try:
        response = requests.get(
            url, timeout=timeout if timeout is not None else HEALTH_TIMEOUT
        )
    except requests.Timeout as exc:
        raise APIError(f"The API did not answer within {timeout or HEALTH_TIMEOUT:.0f} s.") from exc
    except requests.ConnectionError as exc:
        raise APIError(
            f"Cannot reach the API at {normalize_base_url(base_url)}. "
            "Start it with: uvicorn api.main:app --reload"
        ) from exc
    except requests.RequestException as exc:
        raise APIError(f"Health check failed: {exc}") from exc

    if response.status_code >= 400:
        raise APIError(
            f"Health check returned HTTP {response.status_code}: "
            f"{_error_detail(response)}",
            status_code=response.status_code,
        )

    payload = _decode_json(response)
    if not isinstance(payload, dict):
        raise APIError("Health endpoint did not return a JSON object.")
    return payload


def classify_image(
    base_url: str | None,
    image_bytes: bytes,
    filename: str = "image.jpg",
    timeout: float | None = None,
) -> ClassificationResult:
    """Send an image to ``POST /classify`` and parse the result."""
    payload = _post_image(
        base_url, "/classify", image_bytes, filename, timeout
    )
    return ClassificationResult.from_payload(payload)


def detect_tumors(
    base_url: str | None,
    image_bytes: bytes,
    filename: str = "image.jpg",
    timeout: float | None = None,
) -> DetectionResult:
    """Send an image to ``POST /detect`` and parse the result."""
    payload = _post_image(base_url, "/detect", image_bytes, filename, timeout)
    return DetectionResult.from_payload(payload)


def _post_image(
    base_url: str | None,
    endpoint: str,
    image_bytes: bytes,
    filename: str,
    timeout: float | None,
) -> Any:
    if not image_bytes:
        raise APIError("The image is empty.")

    url = f"{normalize_base_url(base_url)}{endpoint}"
    safe_name = Path(filename or "image.jpg").name
    content_type = (
        mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
    )
    effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT

    try:
        response = requests.post(
            url,
            files={"file": (safe_name, image_bytes, content_type)},
            timeout=effective_timeout,
        )
    except requests.Timeout as exc:
        raise APIError(
            f"The API did not answer within {effective_timeout:.0f} s."
        ) from exc
    except requests.ConnectionError as exc:
        raise APIError(
            f"Cannot reach the API at {normalize_base_url(base_url)}. "
            "Start it with: uvicorn api.main:app --reload"
        ) from exc
    except requests.RequestException as exc:
        raise APIError(f"Request to {endpoint} failed: {exc}") from exc

    if response.status_code >= 400:
        raise APIError(
            f"{endpoint} returned HTTP {response.status_code}: "
            f"{_error_detail(response)}",
            status_code=response.status_code,
        )

    return _decode_json(response)


def _decode_json(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        text = (response.text or "").strip()[:300]
        raise APIError(
            f"The API returned invalid JSON: {text or 'empty response'}"
        ) from exc


def _error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return (response.text or "").strip()[:500] or "no details"

    if not isinstance(payload, dict):
        return str(payload)[:500]

    detail = payload.get("detail", payload.get("message", payload))
    if isinstance(detail, list):
        messages = []
        for item in detail:
            if isinstance(item, dict):
                messages.append(str(item.get("msg") or item))
            else:
                messages.append(str(item))
        return "; ".join(messages)[:500]
    return str(detail)[:500]


# ---------------------------------------------------------------------------
# Payload normalisation
# ---------------------------------------------------------------------------
def _parse_probabilities(payload: dict) -> dict[str, float]:
    raw = payload.get("probabilities")
    if raw is None:
        raw = payload.get("scores", payload.get("class_probabilities"))

    probabilities: dict[str, float] = {}

    if isinstance(raw, dict):
        for name, value in raw.items():
            probabilities[str(name)] = _to_float(value)
    elif isinstance(raw, (list, tuple)):
        for index, item in enumerate(raw):
            if isinstance(item, dict):
                name = (
                    item.get("class_name")
                    or item.get("class")
                    or item.get("label")
                    or item.get("name")
                )
                if name is None:
                    name = _class_name_for_index(index)
                value = item.get(
                    "probability",
                    item.get("confidence", item.get("score", item.get("value"))),
                )
                probabilities[str(name)] = _to_float(value)
            else:
                probabilities[_class_name_for_index(index)] = _to_float(item)

    return probabilities


def _class_name_for_index(index: int) -> str:
    if 0 <= index < len(CLASS_NAMES):
        return CLASS_NAMES[index]
    return f"class_{index}"


def _parse_detection(item: Any) -> Detection | None:
    if not isinstance(item, dict):
        return None

    box = item.get("box")
    if box is None:
        box = item.get("bbox", item.get("xyxy"))

    normalized = bool(item.get("normalized", False))

    if isinstance(box, dict):
        x1 = box.get("x1", box.get("xmin", box.get("left")))
        y1 = box.get("y1", box.get("ymin", box.get("top")))
        x2 = box.get("x2", box.get("xmax", box.get("right")))
        y2 = box.get("y2", box.get("ymax", box.get("bottom")))
        normalized = bool(box.get("normalized", normalized))
    elif isinstance(box, (list, tuple)) and len(box) >= 4:
        x1, y1, x2, y2 = box[:4]
    elif isinstance(box, str):
        values = [value for value in box.replace(",", " ").split() if value]
        if len(values) < 4:
            return None
        x1, y1, x2, y2 = values[:4]
    elif all(key in item for key in ("x1", "y1", "x2", "y2")):
        x1, y1, x2, y2 = (
            item["x1"],
            item["y1"],
            item["x2"],
            item["y2"],
        )
    else:
        return None

    coordinates = [
        _to_float(value, default=float("nan"))
        for value in (x1, y1, x2, y2)
    ]
    if any(value != value for value in coordinates):  # NaN
        return None

    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        return None

    raw_class = item.get("class")
    class_id = _optional_int(item.get("class_id"))
    if class_id is None and isinstance(raw_class, (int, float)):
        class_id = int(raw_class)
    elif class_id is None and isinstance(raw_class, str) and raw_class.isdigit():
        class_id = int(raw_class)

    class_name = (
        item.get("class_name")
        or item.get("label")
        or item.get("name")
        or (raw_class if isinstance(raw_class, str) and not raw_class.isdigit() else None)
    )
    if class_name is None and class_id is not None:
        class_name = _class_name_for_index(class_id)
    class_name = str(class_name or "Tumor")

    confidence = item.get(
        "confidence", item.get("score", item.get("probability", 0.0))
    )

    return Detection(
        class_name=class_name,
        confidence=_clamp(_to_float(confidence)),
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        class_id=class_id,
        normalized=normalized,
    )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_detections(
    image: Image.Image,
    detections: Iterable[Detection],
) -> Image.Image:
    """Return a copy of ``image`` with coloured boxes and labels."""
    canvas = image.convert("RGB").copy()
    detections = tuple(detections)
    if not detections:
        return canvas

    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    line_width = max(2, round(min(width, height) * 0.005))
    font_size = max(12, round(min(width, height) * 0.03))
    font = _load_font(font_size)

    for detection in detections:
        x1, y1, x2, y2 = detection.scaled_box(width, height)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        color = CLASS_COLORS.get(detection.class_name, FALLBACK_COLOR)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        label = f"{detection.class_name} {detection.confidence:.0%}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        padding = max(2, font_size // 6)

        label_x = x1
        label_y = y1 - text_height - 2 * padding
        if label_y < 0:
            label_y = y1 + line_width
        label_x = max(0, min(label_x, width - text_width - 2 * padding))

        draw.rectangle(
            (
                label_x,
                label_y,
                label_x + text_width + 2 * padding,
                label_y + text_height + 2 * padding,
            ),
            fill=color,
        )
        draw.text(
            (label_x + padding, label_y + padding - text_box[1]),
            label,
            fill="white",
            font=font,
        )

    return canvas


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Small conversion helpers
# ---------------------------------------------------------------------------
def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
