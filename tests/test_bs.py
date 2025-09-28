"""Tests covering balance sheet extraction helpers."""

from __future__ import annotations

from typing import Iterable, List

from extract.bs import extract_gubitak_iznad_visine_kapitala
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
    row_list = list(rows)
    page = OcrPage(
        page_number=1,
        text="\n".join(row["text"] for row in row_list),
        tsv=row_list,
    )
    return OcrResult(pdf_path="dummy.pdf", pdf_hash="hash", text=page.text, pages=[page])


def _capital_rows(value: str) -> List[dict]:
    return [
        _word(text="Текућа", left=520, top=40, line=1, word_num=1),
        _word(text="година", left=600, top=40, line=1, word_num=2),
        _word(text="A.", left=120, top=140, line=2, word_num=1),
        _word(text="КАПИТАЛ", left=180, top=140, line=2, word_num=2),
        _word(text="AOP", left=280, top=140, line=2, word_num=3),
        _word(text="0401", left=360, top=140, line=2, word_num=4),
        _word(text=value, left=520, top=140, line=2, word_num=5),
    ]


def test_extract_gubitak_uses_negative_capital():
    result = extract_gubitak_iznad_visine_kapitala(
        _result_from_rows(_capital_rows("-123")),
        year_preference="current",
    )

    assert result.success
    assert result.value == 123
    diagnostic = next(
        message
        for message in result.warnings
        if message.code == "diagnostic_value_source"
    )
    assert diagnostic.context["source"] == "capital_formula"
    assert diagnostic.context["aop_code"] == "0401"
    assert diagnostic.context["capital_value"] == -123
    assert diagnostic.context["computed_value"] == 123


def test_extract_gubitak_returns_zero_for_positive_capital():
    result = extract_gubitak_iznad_visine_kapitala(
        _result_from_rows(_capital_rows("456")),
        year_preference="current",
    )

    assert result.success
    assert result.value == 0
    diagnostic = next(
        message
        for message in result.warnings
        if message.code == "diagnostic_value_source"
    )
    assert diagnostic.context["source"] == "capital_formula"
    assert diagnostic.context["aop_code"] == "0401"
    assert diagnostic.context["capital_value"] == 456
    assert diagnostic.context["computed_value"] == 0

