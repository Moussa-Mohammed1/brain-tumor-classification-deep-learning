"""BrainScan REST API.

Expose the trained models over HTTP for the Streamlit interface (or any other
client).

Run it with::

    uvicorn api.main:app --reload --port 8000

Endpoints
---------
``GET /health``
    Service status and model availability.

``POST /classify``
    InceptionV3 tumour classification (multipart field ``file``).

``POST /detect``
    YOLOv8 tumour detection (multipart field ``file``).

Interactive documentation is available at ``/docs``.
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.inference import (
    MAX_IMAGE_BYTES,
    ClassificationPredictor,
    DetectionPredictor,
    ImageTooLargeError,
    InvalidImageError,
    ModelUnavailableError,
    load_image,
)
from api.schemas import (
    ClassificationResponse,
    DetectionResponse,
    HealthResponse,
)

logger = logging.getLogger("brainscan.api")

API_VERSION = "1.0.0"
API_TITLE = "BrainScan API"
API_DESCRIPTION = (
    "Brain tumour classification (InceptionV3) and detection (YOLOv8) "
    "from MRI scans."
)

UPLOAD_DESCRIPTION = (
    "MRI image (JPG, PNG, BMP or WebP) up to "
    f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB."
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
def get_classification_predictor(request: Request) -> ClassificationPredictor:
    return request.app.state.classification_predictor


def get_detection_predictor(request: Request) -> DetectionPredictor:
    return request.app.state.detection_predictor


def _read_upload(file: UploadFile):
    """Read an ``UploadFile`` and return a validated RGB image."""
    data = file.file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageTooLargeError(
            f"The image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit."
        )
    return load_image(
        data,
        filename=file.filename,
        content_type=file.content_type,
    )


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ImageTooLargeError):
        return HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, InvalidImageError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app(
    classification_predictor: ClassificationPredictor | None = None,
    detection_predictor: DetectionPredictor | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Predictors can be injected for tests; by default the service resolves the
    model paths from the project (or the ``BRAINSCAN_*_MODEL`` variables).
    """
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
    )

    app.state.classification_predictor = (
        classification_predictor or ClassificationPredictor()
    )
    app.state.detection_predictor = (
        detection_predictor or DetectionPredictor()
    )

    origins = [
        origin.strip()
        for origin in os.environ.get("BRAINSCAN_CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "name": API_TITLE,
            "version": API_VERSION,
            "docs": "/docs",
            "health": "/health",
            "endpoints": ["/classify", "/detect"],
        }

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["meta"],
        summary="Service and model status",
    )
    def health(request: Request) -> HealthResponse:
        classifier = request.app.state.classification_predictor
        detector = request.app.state.detection_predictor
        models = {
            "classification": bool(classifier.available),
            "detection": bool(detector.available),
        }
        status = "ok" if all(models.values()) else "degraded"
        return HealthResponse(
            status=status,
            models=models,
            version=API_VERSION,
        )

    @app.post(
        "/classify",
        response_model=ClassificationResponse,
        tags=["inference"],
        summary="Classify an MRI scan",
        responses={
            400: {"description": "Invalid image"},
            413: {"description": "Image too large"},
            503: {"description": "Classification model unavailable"},
        },
    )
    def classify(
        file: UploadFile = File(..., description=UPLOAD_DESCRIPTION),
        predictor: ClassificationPredictor = Depends(
            get_classification_predictor
        ),
    ) -> ClassificationResponse:
        try:
            image = _read_upload(file)
        except (ImageTooLargeError, InvalidImageError) as exc:
            raise _http_error(exc) from exc

        started = time.perf_counter()
        try:
            prediction = predictor.predict(image)
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - hide internal details
            logger.exception("Classification failed")
            raise HTTPException(
                status_code=500,
                detail="Classification failed. Check the server logs.",
            ) from exc

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        filename = file.filename or "upload"

        logger.info(
            "classify filename=%s class=%s confidence=%.4f latency_ms=%.2f",
            filename,
            prediction["predicted_class"],
            prediction["confidence"],
            latency_ms,
        )

        return ClassificationResponse(
            filename=filename,
            predicted_class=prediction["predicted_class"],
            confidence=prediction["confidence"],
            probabilities=prediction["probabilities"],
            model=predictor.model_name,
            latency_ms=latency_ms,
        )

    @app.post(
        "/detect",
        response_model=DetectionResponse,
        tags=["inference"],
        summary="Detect and localise tumours",
        responses={
            400: {"description": "Invalid image"},
            413: {"description": "Image too large"},
            503: {"description": "Detection model unavailable"},
        },
    )
    def detect(
        file: UploadFile = File(..., description=UPLOAD_DESCRIPTION),
        predictor: DetectionPredictor = Depends(get_detection_predictor),
    ) -> DetectionResponse:
        try:
            image = _read_upload(file)
        except (ImageTooLargeError, InvalidImageError) as exc:
            raise _http_error(exc) from exc

        started = time.perf_counter()
        try:
            detections = predictor.predict(image)
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - hide internal details
            logger.exception("Detection failed")
            raise HTTPException(
                status_code=500,
                detail="Detection failed. Check the server logs.",
            ) from exc

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        filename = file.filename or "upload"

        logger.info(
            "detect filename=%s detections=%d latency_ms=%.2f",
            filename,
            len(detections),
            latency_ms,
        )

        return DetectionResponse(
            filename=filename,
            detections=detections,
            count=len(detections),
            model=predictor.model_name,
            latency_ms=latency_ms,
            image_width=image.width,
            image_height=image.height,
        )

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.environ.get("BRAINSCAN_API_HOST", "0.0.0.0"),
        port=int(os.environ.get("BRAINSCAN_API_PORT", "8000")),
    )
