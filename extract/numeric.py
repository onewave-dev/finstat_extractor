"""Utilities for normalising numeric strings extracted via OCR."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Iterable, Optional, Tuple

from .models import NumericParseResult


class NumericParseError(ValueError):
    """Raised when a numeric string cannot be normalised."""


def _strip_parentheses(value: str) -> Tuple[str, bool]:
    value = value.strip()
    if value.startswith("(") and value.endswith(")"):
        return value[1:-1], True
    return value, False


def _normalise_sign(value: str) -> Tuple[str, int]:
    sign = 1
    value = value.strip()

    if value.startswith("+"):
        value = value[1:]
    elif value.startswith("-") or value.startswith("\u2212"):
        value = value[1:]
        sign = -1

    value = value.strip()

    if value.endswith("-") or value.endswith("\u2212"):
        value = value[:-1]
        sign *= -1
    elif value.endswith("+"):
        value = value[:-1]

    return value.strip(), sign


def _choose_decimal_separator(value: str) -> Optional[str]:
    positions = {sep: value.rfind(sep) for sep in {",", "."} if sep in value}
    if not positions:
        return None

    # choose the rightmost separator as the decimal candidate
    sep = max(positions, key=lambda item: positions[item])
    pos = positions[sep]
    count = value.count(sep)
    decimals = value[pos + 1 :]

    if count > 1:
        return None
    if len(decimals) == 3 and decimals.isdigit():
        # heuristically assume a thousands group rather than decimals
        return None
    if not decimals:
        return None
    return sep


def _remove_thousands(value: str, separators: Iterable[str]) -> str:
    for sep in separators:
        if sep:
            value = value.replace(sep, "")
    return value


def normalize_numeric_string(
    raw: str,
    *,
    allow_negative: bool = True,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> NumericParseResult:
    """Normalise a numeric string produced by OCR.

    The helper removes thousands separators, converts decimal commas to
    periods (when present) and validates that the resulting value resides
    within the optional ``min_value``/``max_value`` bounds.
    """

    if raw is None:
        raise NumericParseError("No value supplied")

    text = str(raw).strip()
    if not text:
        raise NumericParseError("Empty string")

    text = text.replace("\u00a0", " ")
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    text, was_parenthesised = _strip_parentheses(text)
    text, sign = _normalise_sign(text)
    if was_parenthesised:
        sign *= -1

    text = text.replace(" ", "")
    if not any(ch.isdigit() for ch in text):
        raise NumericParseError(f"No digits detected in '{raw}'")

    decimal_sep = _choose_decimal_separator(text)
    thousands_candidates = {",", "."}
    if decimal_sep:
        thousands_candidates.remove(decimal_sep)

    integer_part, fractional_part = text, None
    if decimal_sep:
        integer_part, fractional_part = text.split(decimal_sep, 1)

    integer_part = _remove_thousands(integer_part, thousands_candidates)
    if fractional_part is not None:
        fractional_part = _remove_thousands(fractional_part, thousands_candidates)

    if not integer_part or not integer_part.isdigit():
        raise NumericParseError(f"Invalid integer component in '{raw}'")
    if fractional_part is not None and not fractional_part.isdigit():
        raise NumericParseError(f"Invalid decimal component in '{raw}'")

    normalized = integer_part
    if fractional_part:
        normalized = f"{integer_part}.{fractional_part}"

    try:
        decimal_value = Decimal(normalized)
    except InvalidOperation as exc:  # pragma: no cover - defensive
        raise NumericParseError(f"Unable to convert '{raw}' to Decimal") from exc

    decimal_value *= sign

    if decimal_value != decimal_value.to_integral_value():
        raise NumericParseError(f"Non-integer value encountered: '{raw}'")

    value = int(decimal_value)
    if value < 0 and not allow_negative:
        raise NumericParseError(f"Negative value is not permitted: {value}")

    if min_value is not None and value < min_value:
        raise NumericParseError(
            f"Value {value} is below the permitted minimum of {min_value}"
        )
    if max_value is not None and value > max_value:
        raise NumericParseError(
            f"Value {value} exceeds the permitted maximum of {max_value}"
        )

    return NumericParseResult(value=value, normalized_text=str(value))


__all__ = ["NumericParseError", "normalize_numeric_string"]

