"""Tests for the OCR engine integration points."""

from __future__ import annotations

import logging
import types

from PIL import Image

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
        {"preprocessing": {"enabled": False}},
        cache_path=tmp_path / "cache.sqlite",
        log_path=tmp_path / "ocr.log",
        logger=logger,
    )

    def fake_render(self, _pdf_path, dpi_override=None):  # pragma: no cover
        return []

    def fake_run(
        self, pdf_path, pdf_hash, images, dpi, plumber_pages=None
    ):  # pragma: no cover - helper for test only
        return OcrResult(
            pdf_path=str(pdf_path),
            pdf_hash=pdf_hash,
            text="",
            pages=[],
            average_confidence=80.0,
        )

    monkeypatch.setattr(
        engine, "_render_pdf", types.MethodType(fake_render, engine)
    )
    monkeypatch.setattr(
        engine, "_run_tesseract", types.MethodType(fake_run, engine)
    )
    monkeypatch.setattr(engine, "_extract_pdf_text", lambda path: [])

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
        {"preprocessing": {"enabled": False}},
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
        {"preprocessing": {"enabled": False}},
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


def test_process_uses_pdfplumber_text(monkeypatch, tmp_path):
    pdf_path = _make_dummy_pdf(tmp_path)

    logger = logging.getLogger("test-pdfplumber")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    engine = OCREngine(
        {"preprocessing": {"enabled": False}},
        cache_path=tmp_path / "cache.sqlite",
        log_path=tmp_path / "ocr.log",
        logger=logger,
    )

    image = Image.new("RGB", (10, 10), color="white")

    def fake_render(self, _path, dpi_override=None):  # pragma: no cover - helper
        return [image]

    monkeypatch.setattr(
        engine, "_render_pdf", types.MethodType(fake_render, engine)
    )
    monkeypatch.setattr(engine, "_extract_pdf_text", lambda path: ["PDF TEXT"])

    monkeypatch.setattr(
        "ocr.ocr_engine.pytesseract.image_to_string",
        lambda *args, **kwargs: "OCR TEXT",
    )
    monkeypatch.setattr(
        "ocr.ocr_engine.pytesseract.image_to_data",
        lambda *args, **kwargs: "level\tconf\n1\t85\n",
    )

    result = engine.process(pdf_path)

    assert result.text == "PDF TEXT"
    assert result.text_source == "pdfplumber"
    assert result.pages[0].text == "PDF TEXT"
    assert result.pages[0].confidence == 85.0


def test_process_retries_with_alternative_dpi(monkeypatch, tmp_path):
    pdf_path = _make_dummy_pdf(tmp_path)

    logger = logging.getLogger("test-dpi-retry")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    engine = OCREngine(
        {
            "dpi_retry": [350],
            "confidence_threshold": 75,
            "preprocessing": {"enabled": False},
        },
        cache_path=tmp_path / "cache.sqlite",
        log_path=tmp_path / "ocr.log",
        logger=logger,
    )

    image = Image.new("RGB", (10, 10), color="white")
    render_calls = []

    def fake_render(self, _path, dpi_override=None):  # pragma: no cover
        render_calls.append(dpi_override)
        return [image]

    monkeypatch.setattr(
        engine, "_render_pdf", types.MethodType(fake_render, engine)
    )
    monkeypatch.setattr(engine, "_extract_pdf_text", lambda path: [])

    low_conf_result = OcrResult(
        pdf_path=str(pdf_path),
        pdf_hash="hash",
        text="low",
        pages=[],
        average_confidence=60.0,
    )
    high_conf_result = OcrResult(
        pdf_path=str(pdf_path),
        pdf_hash="hash",
        text="high",
        pages=[],
        average_confidence=82.0,
    )

    def fake_run(self, pdf_path, pdf_hash, images, dpi, plumber_pages=None):
        return low_conf_result if dpi == 300 else high_conf_result

    monkeypatch.setattr(
        engine, "_run_tesseract", types.MethodType(fake_run, engine)
    )

    result = engine.process(pdf_path)

    assert render_calls == [300, 350]
    assert result.average_confidence == 82.0


def test_ocr_numeric_region_uses_numeric_settings(monkeypatch, tmp_path):
    logger = logging.getLogger("test-numeric")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    engine = OCREngine(
        {"preprocessing": {"enabled": False}},
        cache_path=tmp_path / "cache.sqlite",
        log_path=tmp_path / "ocr.log",
        logger=logger,
    )

    image = Image.new("RGB", (50, 50), color="white")
    captured = {}

    def fake_to_string(img, lang=None, config=None):  # pragma: no cover
        captured["config"] = config
        captured["lang"] = lang
        return "123"

    def fake_to_data(img, lang=None, config=None, output_type=None):  # pragma: no cover
        captured["data_config"] = config
        return "level\tconf\n1\t95\n"

    monkeypatch.setattr("ocr.ocr_engine.pytesseract.image_to_string", fake_to_string)
    monkeypatch.setattr("ocr.ocr_engine.pytesseract.image_to_data", fake_to_data)

    result = engine.ocr_numeric_region(
        image,
        (0, 0, 20, 20),
        {"psm": 8, "preprocessing": {"enabled": False}},
    )

    assert "--oem 1" in captured["config"]
    assert "--psm 8" in captured["config"]
    assert "tessedit_char_whitelist=0123456789()-. ," in captured["config"]
    assert captured["data_config"] == captured["config"]
    assert result["text"] == "123"
    assert result["confidence"] == 95.0
