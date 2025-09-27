"""Extractor helpers for "Биланс стања" documents."""

from __future__ import annotations

from typing import Optional

from ocr.ocr_engine import OcrResult

from .models import ExtractionResult
from .table_extractor import extract_field_from_ocr


def extract_ukupna_aktiva(
    ocr_result: OcrResult,
    *,
    year_preference: Optional[str] = "current",
    min_value: Optional[int] = 0,
    max_value: Optional[int] = None,
) -> ExtractionResult:
    """Extract the "Укупна актива" metric from a BS OCR result."""

    return extract_field_from_ocr(
        ocr_result,
        anchor_key="bs_assets",
        field_name="Укупна актива",
        year_preference=year_preference,
        min_value=min_value,
        max_value=max_value,
    )


def extract_gubitak_iznad_visine_kapitala(
    ocr_result: OcrResult,
    *,
    year_preference: Optional[str] = "current",
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> ExtractionResult:
    """Extract the "Губитак изнад висине капитала" metric from a BS OCR result."""

    return extract_field_from_ocr(
        ocr_result,
        anchor_key="bs_loss",
        field_name="Губитак изнад висине капитала",
        year_preference=year_preference,
        min_value=min_value,
        max_value=max_value,
    )


__all__ = [
    "extract_ukupna_aktiva",
    "extract_gubitak_iznad_visine_kapitala",
]

