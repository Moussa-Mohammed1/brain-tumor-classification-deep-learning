"""Headless smoke tests for the Streamlit interface.

These tests need Streamlit installed and are skipped otherwise. They exercise
the real script with ``streamlit.testing.v1.AppTest``: no API is contacted
because the action buttons are not clicked.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="streamlit is not installed")

from PIL import Image  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app" / "app.py"


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (96, 64), "black").save(buffer, format="PNG")
    return buffer.getvalue()


def test_app_starts_without_exceptions():
    test_app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not test_app.exception
    assert test_app.get("file_uploader")
    assert [button.label for button in test_app.button] == ["Test connection"]


def test_app_renders_the_two_tasks_after_upload():
    test_app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not test_app.exception

    test_app.get("file_uploader")[0].upload(
        "scan.png", _png_bytes(), "image/png"
    ).run()

    assert not test_app.exception
    assert [tab.label for tab in test_app.tabs] == [
        "🧬 Classification",
        "🎯 Detection & localisation",
    ]
    assert {
        button.label for button in test_app.button
    } == {"Test connection", "Classify image", "Detect tumours"}
