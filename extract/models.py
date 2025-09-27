"""Data models shared across extraction modules.

The extractor is intentionally verbose about diagnostics – callers need
actionable information to populate the final report for each Matični broj.
This module keeps the small dataclasses that encode the value returned by
the specialised extractors together with error/warning metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExtractionMessage:
    """Represents a diagnostic message emitted during extraction."""

    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Outcome produced by a field extractor."""

    field_name: str
    value: Optional[int] = None
    raw_text: Optional[str] = None
    normalized_text: Optional[str] = None
    page_number: Optional[int] = None
    anchor_text: Optional[str] = None
    anchor_bbox: Optional[Tuple[int, int, int, int]] = None
    column_label: Optional[str] = None
    column_bbox: Optional[Tuple[int, int, int, int]] = None
    errors: List[ExtractionMessage] = field(default_factory=list)
    warnings: List[ExtractionMessage] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Return ``True`` when a value was extracted without errors."""

        return self.value is not None and not self.errors

    def add_error(self, code: str, message: str, **context: Any) -> None:
        self.errors.append(ExtractionMessage(code=code, message=message, context=context))

    def add_warning(self, code: str, message: str, **context: Any) -> None:
        self.warnings.append(
            ExtractionMessage(code=code, message=message, context=context)
        )


@dataclass
class NumericParseResult:
    """Result of normalising a numeric string."""

    value: int
    normalized_text: str


__all__ = ["ExtractionMessage", "ExtractionResult", "NumericParseResult"]

