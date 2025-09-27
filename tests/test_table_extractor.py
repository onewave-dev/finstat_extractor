"""Synthetic TSV fixtures for :mod:`extract.table_extractor`."""

from __future__ import annotations

from typing import Iterable, List

from extract.table_extractor import extract_field_from_ocr
from ocr.ocr_engine import OcrPage, OcrResult


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


def _result_from_rows(rows: Iterable[dict]) -> OcrResult:
    page = OcrPage(page_number=1, text="\n".join(row["text"] for row in rows), tsv=list(rows))
    return OcrResult(pdf_path="dummy.pdf", pdf_hash="hash", text=page.text, pages=[page])


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