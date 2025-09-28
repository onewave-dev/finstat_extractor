"""Tests for the OCR engine integration points."""

from __future__ import annotations

import logging
import types

from ocr.ocr_engine import DEFAULT_PAGE_LIMIT, OCREngine, OcrResult


def _make_dummy_pdf(tmp_path):
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    return pdf_path


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


def test_render_pdf_applies_default_limit(monkeypatch, tmp_path):
    pdf_path = _make_dummy_pdf(tmp_path)

    logger = logging.getLogger("test-default-limit")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    engine = OCREngine(
        {},
        cache_path=tmp_path / "cache.sqlite",
        log_path=tmp_path / "ocr.log",
        logger=logger,
    )

    monkeypatch.setattr(
        "ocr.ocr_engine.pdfinfo_from_path",
        lambda *args, **kwargs: {"Pages": DEFAULT_PAGE_LIMIT + 10},
    )

    captured_kwargs = {}

    def fake_convert(_path, **kwargs):
        captured_kwargs.update(kwargs)
        return []

    monkeypatch.setattr("ocr.ocr_engine.convert_from_path", fake_convert)

    engine._render_pdf(pdf_path)

    assert captured_kwargs.get("last_page") == DEFAULT_PAGE_LIMIT


def test_render_pdf_logs_message_when_truncated(monkeypatch, tmp_path, caplog):
    pdf_path = _make_dummy_pdf(tmp_path)

    logger = logging.getLogger("test-truncated-limit")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    engine = OCREngine(
        {},
        cache_path=tmp_path / "cache.sqlite",
        log_path=tmp_path / "ocr.log",
        logger=logger,
    )

    monkeypatch.setattr(
        "ocr.ocr_engine.pdfinfo_from_path",
        lambda *args, **kwargs: {"Pages": DEFAULT_PAGE_LIMIT + 5},
    )

    captured_kwargs = {}

    def fake_convert(_path, **kwargs):
        captured_kwargs.update(kwargs)
        return []

    monkeypatch.setattr("ocr.ocr_engine.convert_from_path", fake_convert)

    caplog.set_level(logging.WARNING, logger=logger.name)

    engine._render_pdf(pdf_path)

    assert captured_kwargs.get("last_page") == DEFAULT_PAGE_LIMIT
    assert any(
        "processed only the first" in message.lower()
        and str(DEFAULT_PAGE_LIMIT) in message
        for message in caplog.messages
    )
