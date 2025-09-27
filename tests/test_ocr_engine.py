"""Tests for the OCR engine integration points."""

from __future__ import annotations

import logging
import types

from ocr.ocr_engine import OCREngine, OcrResult


def test_process_logs_cached_message(monkeypatch, tmp_path, caplog):
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    logger = logging.getLogger("test-ocr-engine")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    engine = OCREngine(
        {},
        cache_path=tmp_path / "cache.sqlite",
        log_path=tmp_path / "ocr.log",
        logger=logger,
    )

    def fake_render(self, _pdf_path):  # pragma: no cover - helper for test only
        return []

    def fake_run(self, pdf_path, pdf_hash, images):  # pragma: no cover - helper for test only
        return OcrResult(pdf_path=str(pdf_path), pdf_hash=pdf_hash, text="", pages=[])

    monkeypatch.setattr(
        engine, "_render_pdf", types.MethodType(fake_render, engine)
    )
    monkeypatch.setattr(
        engine, "_run_tesseract", types.MethodType(fake_run, engine)
    )

    caplog.set_level(logging.INFO, logger=logger.name)

    engine.process(pdf_path)
    caplog.clear()

    engine.process(pdf_path)

    assert any("Using cached OCR" in message for message in caplog.messages)
