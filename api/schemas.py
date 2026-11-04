"""Pydantic schemas for the BrainScan FastAPI service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Service and model availability."""

    status: str = Field(description="'ok' when both models are available")
    models: dict[str, bool] = Field(
        description="Availability of the classification and detection models"
    )
    version: str = Field(description="API version")


class ClassificationResponse(BaseModel):
    """Response of ``POST /classify``."""

    filename: str = Field(description="Original upload filename")
    predicted_class: str = Field(description="Winning class name")
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(
        description="Per-class probability, in the training class order"
    )
    model: str = Field(description="Model identifier")
    latency_ms: float = Field(ge=0.0, description="Server inference time")


class BoundingBox(BaseModel):
    """Pixel coordinates in the original image (``xyxy``)."""

    x1: float
    y1: float
    x2: float
    y2: float


class DetectionItem(BaseModel):
    """One detected tumour region."""

    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: BoundingBox


class DetectionResponse(BaseModel):
    """Response of ``POST /detect``."""

    filename: str = Field(description="Original upload filename")
    detections: list[DetectionItem] = Field(default_factory=list)
    count: int = Field(ge=0, description="Number of returned detections")
    model: str = Field(description="Model identifier")
    latency_ms: float = Field(ge=0.0, description="Server inference time")
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
