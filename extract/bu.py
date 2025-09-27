"""Extractor for "Биланс успеха" specific metrics."""

from __future__ import annotations

from typing import Optional

from ocr.ocr_engine import OcrResult

from .models import ExtractionResult
from .table_extractor import extract_field_from_ocr


def extract_poslovni_prihodi(
    ocr_result: OcrResult,
    *,
    year_preference: Optional[str] = "current",
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> ExtractionResult:
    """Extract the "Пословни приходи" value from a BU OCR result."""

    return extract_field_from_ocr(
        ocr_result,
        anchor_key="bu_revenue",
        field_name="Пословни приходи",
        year_preference=year_preference,
        min_value=min_value,
        max_value=max_value,
    )


__all__ = ["extract_poslovni_prihodi"]

