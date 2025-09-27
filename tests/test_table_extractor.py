"""Synthetic TSV fixtures for :mod:`extract.table_extractor`."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import pytest

from extract.table_extractor import (
    OcrLine,
    OcrWord,
    _detect_year_columns,
    extract_field_from_ocr,
)
from ocr.ocr_engine import OcrPage, OcrResult
from ocr import anchors


def _word(
    *,
    text: str,
    left: int,
    top: int,
    width: int = 60,
    height: int = 20,
    page: int = 1,
    block: int = 1,
    par: int = 1,
    line: int = 1,
    word_num: int,
) -> dict:
    return {
        "level": "5",
        "page_num": str(page),
        "block_num": str(block),
        "par_num": str(par),
        "line_num": str(line),
        "word_num": str(word_num),
        "left": str(left),
        "top": str(top),
        "width": str(width),
        "height": str(height),
        "conf": "95",
        "text": text,
    }


def _ocr_word(
    text: str,
    *,
    left: int,
    top: int,
    width: int = 60,
    height: int = 20,
) -> OcrWord:
    return OcrWord(
        text=text,
        left=left,
        top=top,
        width=width,
        height=height,
        raw={
            "text": text,
            "left": str(left),
            "top": str(top),
            "width": str(width),
            "height": str(height),
        },
    )


def _result_from_rows(rows: Iterable[dict]) -> OcrResult:
    page = OcrPage(page_number=1, text="\n".join(row["text"] for row in rows), tsv=list(rows))
    return OcrResult(pdf_path="dummy.pdf", pdf_hash="hash", text=page.text, pages=[page])


def _make_line(words: Sequence[OcrWord]) -> OcrLine:
    return OcrLine(page_number=1, block_num=1, par_num=1, line_num=1, words=list(words))


def test_extract_field_prefers_numeric_cluster_right_of_anchor():
    rows: List[dict] = []
    # Year header to detect the "current" column.
    rows.append(
        _word(text="Текућа", left=100, top=50, line=1, word_num=1)
    )
    rows.append(
        _word(text="година", left=180, top=50, line=1, word_num=2)
    )

    # Numeric token to the left of the anchor that should be ignored.
    rows.append(
        _word(text="999", left=40, top=150, line=2, word_num=1)
    )
    rows.append(
        _word(text="Пословни", left=120, top=150, line=2, word_num=2)
    )
    rows.append(
        _word(text="приходи", left=220, top=150, line=2, word_num=3)
    )
    rows.append(
        _word(text="123", left=340, top=150, line=2, word_num=4)
    )
    rows.append(
        _word(text="456", left=420, top=150, line=2, word_num=5)
    )

    result = extract_field_from_ocr(
        _result_from_rows(rows),
        anchor_key="bu_revenue",
        field_name="revenue",
        year_preference="current",
    )

    assert result.success
    assert result.value == 123456
    assert result.column_label == "current"
    assert not result.errors


def test_extract_field_uses_left_cluster_when_no_right_candidate():
    rows: List[dict] = []
    rows.append(
        _word(text="Текућа", left=100, top=50, line=1, word_num=1)
    )
    rows.append(
        _word(text="година", left=180, top=50, line=1, word_num=2)
    )
    rows.append(
        _word(text="123", left=40, top=150, line=2, word_num=1)
    )
    rows.append(
        _word(text="Пословни", left=120, top=150, line=2, word_num=2)
    )
    rows.append(
        _word(text="приходи", left=220, top=150, line=2, word_num=3)
    )

    result = extract_field_from_ocr(
        _result_from_rows(rows),
        anchor_key="bu_revenue",
        field_name="revenue",
        year_preference="current",
    )

    assert result.success
    assert result.value == 123
    assert result.column_label == "current"


def test_extract_field_falls_back_to_previous_year_when_preferred_missing():
    rows: List[dict] = []
    rows.append(
        _word(text="Претходна", left=100, top=50, line=1, word_num=1)
    )
    rows.append(
        _word(text="година", left=210, top=50, line=1, word_num=2)
    )
    rows.append(
        _word(text="Пословни", left=120, top=150, line=2, word_num=3)
    )
    rows.append(
        _word(text="приходи", left=220, top=150, line=2, word_num=4)
    )
    rows.append(
        _word(text="321", left=360, top=150, line=2, word_num=5)
    )

    result = extract_field_from_ocr(
        _result_from_rows(rows),
        anchor_key="bu_revenue",
        field_name="revenue",
        year_preference="current",
    )

    assert result.success
    assert result.value == 321
    assert result.column_label == "previous"
    assert any(msg.code == "column_fallback_used" for msg in result.warnings)


@pytest.mark.parametrize(
    "words",
    [
        [("Текућа", 100), ("година", 180)],
        [("Текуће", 100), ("године", 200)],
        [("tekuca", 100), ("godina", 180)],
    ],
)
def test_detect_year_columns_handles_textual_variants(monkeypatch, words):
    reference_year = 2024
    monkeypatch.setattr(anchors, "get_reference_year", lambda: reference_year)
    line = _make_line(
        [
            _ocr_word(text, left=left, top=40)
            for text, left in words
        ]
    )

    columns = _detect_year_columns([line])

    assert "current" in columns
    assert columns["current"].label == "current"
    assert columns["current"].left == min(left for _, left in words)


@pytest.mark.parametrize("reference_year", [2024, 2025])
def test_detect_year_columns_handles_numeric_years(monkeypatch, reference_year):
    monkeypatch.setattr(anchors, "get_reference_year", lambda: reference_year)
    line = _make_line(
        [
            _ocr_word(str(reference_year - 1), left=80, top=30),
            _ocr_word(str(reference_year), left=200, top=30),
        ]
    )

    columns = _detect_year_columns([line])

    assert columns["current"].label == "current"
    assert columns["current"].left == 200
    assert columns["previous"].label == "previous"
    assert columns["previous"].left == 80
