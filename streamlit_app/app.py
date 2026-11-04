"""BrainScan — Streamlit interface for brain-tumour analysis.

Run it with::

    streamlit run streamlit_app/app.py

The UI never loads TensorFlow or Ultralytics itself. It uploads the image to
the FastAPI service over HTTP, then renders the JSON response:

* ``POST /classify`` → predicted class + per-class confidence scores,
* ``POST /detect``   → bounding boxes drawn on the original image.

Set ``BRAINSCAN_API_URL`` (default ``http://127.0.0.1:8000``) or edit the URL
in the sidebar when the API runs elsewhere.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.api_client import (  # noqa: E402
    APIError,
    CLASS_NAMES,
    ClassificationResult,
    DetectionResult,
    check_health,
    classify_image,
    default_api_url,
    detect_tumors,
    draw_detections,
)

ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "bmp"]
MAX_UPLOAD_MB = 15

PAGE_TITLE = "BrainScan · Brain Tumour Analysis"


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_css()
    api_url = _render_sidebar()

    st.markdown(
        """
        <div class="brainscan-hero">
          <h1>🧠 BrainScan</h1>
          <p>Upload an MRI scan to classify the tumour type and localise
          lesions with the YOLOv8 detector.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "MRI scan",
        type=ALLOWED_EXTENSIONS,
        help=(
            f"Accepted formats: JPG, JPEG, PNG, BMP — up to {MAX_UPLOAD_MB} MB. "
            "The image is sent to the FastAPI service and is not stored."
        ),
    )

    if uploaded is None:
        _render_empty_state()
        st.stop()

    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(
            f"The file is larger than {MAX_UPLOAD_MB} MB. "
            "Please upload a smaller scan."
        )
        st.stop()

    image_bytes = uploaded.getvalue()
    _reset_results_for_new_image(image_bytes)

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("The uploaded file is not a valid image.")
        st.stop()

    _render_image_summary(image, uploaded)

    tab_classification, tab_detection = st.tabs(
        ["🧬 Classification", "🎯 Detection & localisation"]
    )

    with tab_classification:
        _render_classification_tab(api_url, image_bytes, uploaded.name)

    with tab_detection:
        _render_detection_tab(api_url, image, image_bytes, uploaded.name)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def _inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
          .brainscan-hero h1 { margin-bottom: 0.1rem; }
          .brainscan-hero p { color: #6b7280; font-size: 1.05rem; }
          div[data-testid="stMetricValue"] { font-size: 1.7rem; }
          .brainscan-empty {
            border: 1px dashed #c7cbd1;
            border-radius: 0.8rem;
            padding: 2.5rem 1rem;
            text-align: center;
            color: #6b7280;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> str:
    with st.sidebar:
        st.header("⚙️ Configuration")

        previous_url = st.session_state.get("api_url", default_api_url())
        api_url = st.text_input(
            "FastAPI base URL",
            value=previous_url,
            help="Where the FastAPI service listens, e.g. http://127.0.0.1:8000",
        )
        if api_url != previous_url:
            # Results and the health badge belong to the previous endpoint.
            st.session_state.api_health = None
            st.session_state.classification = None
            st.session_state.detection = None
        st.session_state.api_url = api_url

        if st.button("Test connection", width="stretch"):
            with st.spinner("Contacting the API…"):
                try:
                    health = check_health(api_url)
                except APIError as error:
                    st.session_state.api_health = None
                    st.error(str(error))
                else:
                    st.session_state.api_health = health
                    st.success("API online")

        health = st.session_state.get("api_health")
        if isinstance(health, dict):
            st.caption(f"Status: **{health.get('status', 'ok')}**")
            models = health.get("models")
            if isinstance(models, dict) and models:
                for name, available in models.items():
                    icon = "✅" if available else "❌"
                    st.caption(f"{icon} {name}")

        st.divider()
        st.markdown("**Endpoints used**")
        st.code(
            "GET  /health\nPOST /classify\nPOST /detect",
            language="text",
        )
        st.caption(
            "Start the service from the project root:\n\n"
            "`uvicorn api.main:app --reload`"
        )

        st.divider()
        st.markdown("**About**")
        st.caption(
            "InceptionV3 (classification) and YOLOv8 (detection) models "
            "exposed by the BrainScan FastAPI service. Predictions are "
            "decision-support only and do not replace a radiologist."
        )

    return api_url


def _render_empty_state() -> None:
    st.markdown(
        """
        <div class="brainscan-empty">
          <h3>No scan selected</h3>
          <p>Upload a JPG, PNG or BMP MRI image to start the analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Two tasks are available: tumour classification (4 classes) and "
        "tumour detection with bounding boxes."
    )


def _render_image_summary(image: Image.Image, uploaded) -> None:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.image(image, caption=uploaded.name, width="stretch")

    with right:
        st.subheader("Image details")
        details = {
            "File": uploaded.name,
            "Format": image.format or Path(uploaded.name).suffix.lstrip(".").upper(),
            "Dimensions": f"{image.width} × {image.height} px",
            "Size": f"{uploaded.size / 1024:.1f} KB",
        }
        for label, value in details.items():
            st.markdown(f"**{label}:** {value}")

        st.info(
            "Pick a task below. Classification and detection are independent "
            "and can both be run on the same image."
        )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _render_classification_tab(
    api_url: str,
    image_bytes: bytes,
    filename: str,
) -> None:
    st.subheader("Tumour classification")
    st.caption(
        "InceptionV3 predicts one of four classes: Glioma, Meningioma, "
        "No Tumor or Pituitary."
    )

    if st.button(
        "Classify image",
        key="classify_button",
        type="primary",
        width="stretch",
    ):
        with st.spinner("Running the classifier…"):
            try:
                result = classify_image(api_url, image_bytes, filename)
            except APIError as error:
                st.error(str(error))
                if error.detail:
                    st.caption(error.detail)
            else:
                st.session_state.classification = result
                st.session_state.api_health = (
                    st.session_state.get("api_health")
                    or {"status": "ok"}
                )

    result = st.session_state.get("classification")
    if result is None:
        st.caption("Run the classifier to see the predicted class and scores.")
        return

    _render_classification_result(result)


def _render_classification_result(result: ClassificationResult) -> None:
    st.divider()

    first, second = st.columns(2)
    first.metric("Predicted class", result.predicted_class)
    second.metric("Confidence", f"{result.confidence:.2%}")

    if result.model:
        st.caption(f"Model: {result.model}")
    if result.latency_ms is not None:
        st.caption(f"Inference time: {result.latency_ms:.0f} ms")

    st.markdown("#### Class probabilities")

    ordered = {
        name: result.probabilities.get(name, 0.0) for name in CLASS_NAMES
    }
    for name, probability in result.probabilities.items():
        if name not in ordered:
            ordered[name] = probability

    for name, probability in sorted(
        ordered.items(), key=lambda item: item[1], reverse=True
    ):
        st.write(f"**{name}** · {probability:.2%}")
        st.progress(max(0.0, min(1.0, probability)))

    st.bar_chart(ordered, height=220)

    st.download_button(
        "Download classification JSON",
        data=json.dumps(result.to_dict(), indent=2),
        file_name="classification.json",
        mime="application/json",
        key="download_classification",
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _render_detection_tab(
    api_url: str,
    image: Image.Image,
    image_bytes: bytes,
    filename: str,
) -> None:
    st.subheader("Tumour detection & localisation")
    st.caption(
        "YOLOv8 returns bounding boxes with a class and a confidence score. "
        "Boxes are drawn on your image below."
    )

    if st.button(
        "Detect tumours",
        key="detect_button",
        type="primary",
        width="stretch",
    ):
        with st.spinner("Running the detector…"):
            try:
                result = detect_tumors(api_url, image_bytes, filename)
            except APIError as error:
                st.error(str(error))
                if error.detail:
                    st.caption(error.detail)
            else:
                st.session_state.detection = result
                st.session_state.api_health = (
                    st.session_state.get("api_health")
                    or {"status": "ok"}
                )

    result = st.session_state.get("detection")
    if result is None:
        st.caption("Run the detector to see bounding boxes and scores.")
        return

    _render_detection_result(image, result)


def _render_detection_result(
    image: Image.Image,
    result: DetectionResult,
) -> None:
    st.divider()

    count = len(result.detections)
    if count:
        st.success(f"{count} tumour region{'s' if count != 1 else ''} detected.")
        counts = Counter(
            detection.class_name for detection in result.detections
        )
        st.caption(
            " · ".join(
                f"{name}: {number}" for name, number in counts.items()
            )
        )
    else:
        st.warning(
            "No tumour region detected above the model confidence threshold."
        )

    if result.model:
        st.caption(f"Model: {result.model}")
    if result.latency_ms is not None:
        st.caption(f"Inference time: {result.latency_ms:.0f} ms")

    annotated = draw_detections(image, result.detections)

    left, right = st.columns(2, gap="large")
    with left:
        st.image(image, caption="Original", width="stretch")
    with right:
        st.image(annotated, caption="YOLOv8 detections", width="stretch")

    if count:
        rows = [
            {
                "#": index,
                "Class": detection.class_name,
                "Confidence": f"{detection.confidence:.1%}",
                "x1": round(detection.x1, 1),
                "y1": round(detection.y1, 1),
                "x2": round(detection.x2, 1),
                "y2": round(detection.y2, 1),
            }
            for index, detection in enumerate(result.detections, start=1)
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        rows = []

    st.download_button(
        "Download detection JSON",
        data=json.dumps(result.to_dict(), indent=2),
        file_name="detections.json",
        mime="application/json",
        key="download_detection",
        disabled=not rows,
    )


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def _reset_results_for_new_image(image_bytes: bytes) -> None:
    digest = hashlib.sha256(image_bytes).hexdigest()
    if st.session_state.get("image_digest") == digest:
        return
    st.session_state.image_digest = digest
    st.session_state.classification = None
    st.session_state.detection = None


if __name__ == "__main__":
    main()
