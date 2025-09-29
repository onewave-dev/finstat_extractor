"""Tests for dependency validation helpers."""

from __future__ import annotations

import importlib
import logging

import pytest

import app


def test_configure_dependencies_missing_cv2(monkeypatch):
    """A friendly DependencyError is raised when cv2 cannot be imported."""

    original_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("No module named 'cv2'")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(app, "resolve_tesseract_path", lambda value: "tesseract")
    monkeypatch.setattr(app, "resolve_poppler_path", lambda value: "poppler")
    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    config: dict[str, object] = {}
    logger = logging.getLogger("dependency-test")

    with pytest.raises(app.DependencyError) as excinfo:
        app.configure_dependencies(config, logger=logger)

    assert "pip install opencv-python" in str(excinfo.value)
