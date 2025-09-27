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
import re
from typing import Dict, Iterable, List, Pattern

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
}


# ---------------------------------------------------------------------------
# Year column headers
# ---------------------------------------------------------------------------


YEAR_COLUMN_PATTERNS: Dict[str, Pattern[str]] = {
    "current": re.compile(r"(?i)текућ[ае]\s+годин[ае]"),
    "previous": re.compile(r"(?i)претходн[ае]\s+годин[ае]"),
}


YEAR_COLUMN_SYNONYMS: Dict[str, Iterable[str]] = {
    "current": [
        "текућа година",
        "текуће године",
        "текућа г.",
        "tekuca godina",
        "tekuce godine",
        "2023",  # placeholder for context-specific headers
    ],
    "previous": [
        "претходна година",
        "претходне године",
        "претходна г.",
        "prethodna godina",
        "prethodne godine",
        "2022",  # placeholder when tables use explicit year labels
    ],
}


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


def build_anchor_map() -> Dict[str, AnchorDefinition]:
    """Create a canonical anchor mapping for downstream modules."""

    anchors: Dict[str, AnchorDefinition] = {}

    for key, pattern in ROW_ANCHOR_PATTERNS.items():
        anchors[key] = AnchorDefinition(
            key=key,
            pattern=pattern,
            synonyms=list(ROW_ANCHOR_SYNONYMS.get(key, [])),
        )

    for key, pattern in YEAR_COLUMN_PATTERNS.items():
        anchors[f"year_{key}"] = AnchorDefinition(
            key=f"year_{key}",
            pattern=pattern,
            synonyms=list(YEAR_COLUMN_SYNONYMS.get(key, [])),
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
    "AnchorDefinition",
    "build_anchor_map",
]

