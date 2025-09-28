"""Helpers dedicated to extracting values from the balance sheet (BS)."""

from __future__ import annotations

from typing import Optional

from ocr.ocr_engine import OcrResult

from .models import ExtractionMessage, ExtractionResult
from .table_extractor import (
    extract_capital_aop0401,
    extract_field_from_ocr,
)


_LOSS_FIELD_NAME = "Губитак изнад висине капитала"
_CAPITAL_AOP_CODE = "0401"


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
    """Return the loss above capital, prioritising the capital formula."""

    capital_result = extract_capital_aop0401(
        ocr_result,
        year_preference=year_preference,
        min_value=min_value,
        max_value=max_value,
    )

    if capital_result.success and capital_result.value is not None:
        computed_value = abs(capital_result.value) if capital_result.value < 0 else 0
        loss_result = _clone_result_with_new_field(capital_result, _LOSS_FIELD_NAME)
        loss_result.value = computed_value
        loss_result.warnings.extend(capital_result.warnings)
        loss_result.warnings.append(
            ExtractionMessage(
                code="diagnostic_value_source",
                message=(
                    "Value derived from capital AOP 0401 using the prescribed formula"
                ),
                context={
                    "source": "capital_formula",
                    "aop_code": _CAPITAL_AOP_CODE,
                    "capital_value": capital_result.value,
                    "computed_value": computed_value,
                },
            )
        )
        return loss_result

    fallback_result = extract_field_from_ocr(
        ocr_result,
        anchor_key="bs_loss",
        field_name=_LOSS_FIELD_NAME,
        year_preference=year_preference,
        min_value=min_value,
        max_value=max_value,
    )

    if fallback_result.success and fallback_result.value is not None:
        if capital_result.errors:
            fallback_result.warnings.append(
                ExtractionMessage(
                    code="capital_extraction_failed",
                    message="Capital based derivation failed; using direct loss row",
                    context={
                        "attempted_source": "capital_formula",
                        "aop_code": _CAPITAL_AOP_CODE,
                        "errors": [msg.code for msg in capital_result.errors],
                    },
                )
            )
        fallback_result.warnings.extend(capital_result.warnings)
        fallback_result.warnings.append(
            ExtractionMessage(
                code="diagnostic_value_source",
                message="Value read directly from the loss row (AOP 0401)",
                context={
                    "source": "direct_row",
                    "aop_code": _CAPITAL_AOP_CODE,
                    "computed_value": fallback_result.value,
                },
            )
        )
        return fallback_result

    return _combine_failed_results(capital_result, fallback_result)


def _clone_result_with_new_field(
    source: ExtractionResult, field_name: str
) -> ExtractionResult:
    clone = ExtractionResult(field_name=field_name)
    clone.raw_text = source.raw_text
    clone.normalized_text = source.normalized_text
    clone.page_number = source.page_number
    clone.anchor_text = source.anchor_text
    clone.anchor_bbox = source.anchor_bbox
    clone.column_label = source.column_label
    clone.column_bbox = source.column_bbox
    return clone


def _combine_failed_results(
    capital_result: ExtractionResult, fallback_result: ExtractionResult
) -> ExtractionResult:
    combined = ExtractionResult(field_name=_LOSS_FIELD_NAME)
    combined.errors.extend(capital_result.errors)
    combined.errors.extend(fallback_result.errors)
    combined.warnings.extend(capital_result.warnings)
    combined.warnings.extend(fallback_result.warnings)
    if not combined.errors:
        combined.add_error(
            "value_not_found",
            "Neither the capital formula nor the direct loss row produced a value",
            source_attempts=["capital_formula", "direct_row"],
            aop_code=_CAPITAL_AOP_CODE,
        )
    return combined


__all__ = [
    "extract_ukupna_aktiva",
    "extract_gubitak_iznad_visine_kapitala",
]

