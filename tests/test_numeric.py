"""Unit tests for :mod:`extract.numeric`."""

import pytest

from extract.numeric import NumericParseError, normalize_numeric_string


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.234.567", 1234567),
        ("1 234 567", 1234567),
        ("1.234,00", 1234),
        ("000123", 123),
    ],
)
def test_normalize_numeric_string_handles_thousand_separators(raw, expected):
    result = normalize_numeric_string(raw)
    assert result.value == expected
    assert result.normalized_text == str(expected)


def test_normalize_numeric_string_understands_parentheses_as_negative():
    result = normalize_numeric_string("(12 345)")
    assert result.value == -12345
    assert result.normalized_text == "-12345"


@pytest.mark.parametrize("raw", ["abc", "", None])
def test_normalize_numeric_string_rejects_invalid_inputs(raw):
    with pytest.raises(NumericParseError):
        normalize_numeric_string(raw)  # type: ignore[arg-type]


def test_normalize_numeric_string_respects_bounds_and_sign():
    with pytest.raises(NumericParseError):
        normalize_numeric_string("-1", allow_negative=False)

    with pytest.raises(NumericParseError):
        normalize_numeric_string("9", min_value=10)

    with pytest.raises(NumericParseError):
        normalize_numeric_string("11", max_value=10)