"""Utility definitions for locating anchors within OCR output.

This module centralises all regular expressions and synonym mappings that
are required to detect key Serbian financial statement terms in OCR text.

The expressions are intentionally permissive – they accept minor spelling
variations, casing differences and a mix of Cyrillic/Latin scripts.  The
structures defined here are consumed by the OCR/indexing and extraction
modules to match rows, headers and Matični broj identifiers reliably.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Dict, Iterable, List, Optional, Pattern

# ---------------------------------------------------------------------------
# Matični broj detection
# ---------------------------------------------------------------------------


MAT_BR_PATTERN: Pattern[str] = re.compile(
    r"""
    (?:ма[тт]ични|maticni|matični)          # word "Matični" with variants
    \s*                                     # optional whitespace
    (?:број|broj)                            # word "broj" in Cyrillic/Latin
    \s*[:：\-\u2013]?\s*                   # optional delimiter
    (?P<number>\d{8,9})                     # capture the numeric identifier
    """,
    re.IGNORECASE | re.VERBOSE,
)


MB_INLINE_PATTERN: Pattern[str] = re.compile(r"(?<!\d)(\d{8,9})(?!\d)")


MB_SYNONYMS: Dict[str, Iterable[str]] = {
    "maticni_broj": [
        "матични број",
        "мат. број",
        "матични бр",
        "maticni broj",
        "matični broj",
        "mat. broj",
        "MB",
    ],
}


# ---------------------------------------------------------------------------
# Form header detection
# ---------------------------------------------------------------------------


FORM_HEADER_PATTERNS: Dict[str, Pattern[str]] = {
    "bu": re.compile(r"(?i)билан[сc]\s+успеха|bilans\s+uspeha"),
    "bs": re.compile(r"(?i)билан[сc]\s+стања|bilans\s+stanja"),
}


FORM_HEADER_SYNONYMS: Dict[str, Iterable[str]] = {
    "bu": [
        "биланс успеха",
        "биланс успjеха",
        "биланc успеха",
        "bilans uspeha",
        "bilans uspjeha",
        "bilans uspieha",
        "bu",
    ],
    "bs": [
        "биланс стања",
        "биланс стања",
        "биланc стања",
        "bilans stanja",
        "bilans stanje",
        "bs",
    ],
}


# ---------------------------------------------------------------------------
# Row anchors (target metrics)
# ---------------------------------------------------------------------------


ROW_ANCHOR_PATTERNS: Dict[str, Pattern[str]] = {
    "bu_revenue": re.compile(r"(?i)пословни\s+приходи"),
    "bs_assets": re.compile(r"(?i)укупна\s+актива"),
    "bs_loss": re.compile(r"(?i)губитак\s+изнад\s+висине\s+капитала"),
    "bs_capital_aop0401": re.compile(
        r"""
        (?:
            \bA\s*\.\s*капитал\b.*?0*401 |   # "A. КАПИТАЛ" row referencing AOP
            \bAOP\s*0*401\b |                 # Latin AOP notation
            \bАОП\s*0*401\b                   # Cyrillic AOP notation
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
}


ROW_ANCHOR_SYNONYMS: Dict[str, Iterable[str]] = {
    "bu_revenue": [
        "пословни приходи",
        "poslovni prihodi",
        "приходи из пословања",
    ],
    "bs_assets": [
        "укупна актива",
        "ukupna aktiva",
        "укупна актива (у 000 рсд)",
    ],
    "bs_loss": [
        "губитак изнад висине капитала",
        "gubitak iznad visine kapitala",
        "губитак изнад висине капитала (у 000 рсд)",
    ],
    "bs_capital_aop0401": [
        "a. капитал",
        "а. капитал",
        "a капитал",
        "a. kapital",
        "a kapital",
        "а капитал",
        "капитал (аоп 0401)",
        "капитал аоп 0401",
        "капитал aop 0401",
        "capital aop 0401",
        "kapital aop 0401",
        "aop 0401",
        "аоп 0401",
        "aop0401",
        "a. capital",
    ],
}


# ---------------------------------------------------------------------------
# Year column headers
# ---------------------------------------------------------------------------


YEAR_COLUMN_PATTERNS: Dict[str, Pattern[str]] = {
    "current": re.compile(r"(?i)текућ[ае]\s+годин[ае]"),
    "previous": re.compile(r"(?i)претходн[ае]\s+годин[ае]"),
}


_YEAR_COLUMN_TEXTUAL_SYNONYMS: Dict[str, Iterable[str]] = {
    "current": [
        "текућа година",
        "текуће године",
        "текућа г.",
        "tekuca godina",
        "tekuce godine",
    ],
    "previous": [
        "претходна година",
        "претходне године",
        "претходна г.",
        "prethodna godina",
        "prethodne godine",
    ],
}


YEAR_FOUR_DIGIT_PATTERN: Pattern[str] = re.compile(r"(?<!\d)(?P<year>\d{4})(?!\d)")


def _resolve_reference_year(reference_year: Optional[int]) -> int:
    return reference_year if reference_year is not None else date.today().year


def get_reference_year(reference_year: Optional[int] = None) -> int:
    """Return the baseline "current" year for column detection."""

    return _resolve_reference_year(reference_year)


def get_year_column_synonyms(
    reference_year: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Return textual and numeric variants for year column headers."""

    year = _resolve_reference_year(reference_year)
    synonyms: Dict[str, List[str]] = {
        key: list(values) for key, values in _YEAR_COLUMN_TEXTUAL_SYNONYMS.items()
    }
    synonyms.setdefault("current", []).append(str(year))
    synonyms.setdefault("previous", []).append(str(year - 1))
    return synonyms


YEAR_COLUMN_SYNONYMS: Dict[str, Iterable[str]] = get_year_column_synonyms()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorDefinition:
    """Represents a canonical anchor with regex and synonym spellings."""

    key: str
    pattern: Pattern[str]
    synonyms: List[str]

    def matches(self, text: str) -> bool:
        """Return ``True`` when the regex detects the anchor in ``text``."""

        return bool(self.pattern.search(text))

    def any_synonym_in(self, text: str) -> bool:
        """Check whether any synonym is present inside the provided text."""

        lowered = text.lower()
        return any(syn.lower() in lowered for syn in self.synonyms)


def build_anchor_map(reference_year: Optional[int] = None) -> Dict[str, AnchorDefinition]:
    """Create a canonical anchor mapping for downstream modules."""

    anchors: Dict[str, AnchorDefinition] = {}

    for key, pattern in ROW_ANCHOR_PATTERNS.items():
        anchors[key] = AnchorDefinition(
            key=key,
            pattern=pattern,
            synonyms=list(ROW_ANCHOR_SYNONYMS.get(key, [])),
        )

    resolved_year = get_reference_year(reference_year)
    year_synonyms = get_year_column_synonyms(reference_year=resolved_year)
    for key, pattern in YEAR_COLUMN_PATTERNS.items():
        anchors[f"year_{key}"] = AnchorDefinition(
            key=f"year_{key}",
            pattern=pattern,
            synonyms=list(year_synonyms.get(key, [])),
        )

    return anchors


__all__ = [
    "MAT_BR_PATTERN",
    "MB_INLINE_PATTERN",
    "MB_SYNONYMS",
    "FORM_HEADER_PATTERNS",
    "FORM_HEADER_SYNONYMS",
    "ROW_ANCHOR_PATTERNS",
    "ROW_ANCHOR_SYNONYMS",
    "YEAR_COLUMN_PATTERNS",
    "YEAR_COLUMN_SYNONYMS",
    "YEAR_FOUR_DIGIT_PATTERN",
    "AnchorDefinition",
    "get_reference_year",
    "get_year_column_synonyms",
    "build_anchor_map",
]

