"""PDF index construction utilities.

This module is responsible for walking the current working directory,
running OCR on every ``*.pdf`` file that is found and classifying the
document using the heuristics described in §2.2 of the project spec.

The resulting structure groups files by detected ``Matični broj`` (MB)
and by form type (``Bilans uspeha``/``Bilans stanja``).  Consumers can
either iterate over the full structure or call :func:`get_pdf_for` to
retrieve the most relevant document for a given MB and form type.  The
index keeps track of potential issues (missing MBs, ambiguous form
types, duplicates, etc.) so they can be surfaced in diagnostic reports
later in the pipeline.

The implementation intentionally tolerates partially implemented
``ocr_engine``/``anchors`` modules.  When those modules expose richer
helpers (``detect_maticni_broj``/``detect_form_type``/etc.) they are
used; otherwise the module falls back to conservative regular-expression
based detection so that tests can provide simple stubs.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Tuple, Union

from ocr import anchors, ocr_engine


LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures


FORM_TYPES = {"bu", "bs"}


@dataclass(frozen=True)
class PeriodInfo:
    """Represents a period detected inside the PDF.

    Attributes
    ----------
    label:
        Canonical label such as ``"current"`` / ``"previous"``.  The
        value is normalised to lowercase when the instance is created.
    year:
        Four digit year if it could be extracted from the document.
    raw:
        Original object returned by :mod:`anchors`.  Kept for debugging
        and for potential future enrichments.
    """

    label: Optional[str] = None
    year: Optional[int] = None
    raw: Optional[object] = None

    def __post_init__(self) -> None:  # pragma: no cover - tiny helper
        if self.label is not None and getattr(self, "label", None) != self.label:
            # dataclasses with ``frozen=True`` require object.__setattr__.
            object.__setattr__(self, "label", self.label.lower())


@dataclass
class IndexedPdf:
    """Entry representing a single PDF file inside the index."""

    path: Path
    mb: str
    form_type: str
    mtime: float
    period: Optional[PeriodInfo] = None
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mb = normalize_mb(self.mb)
        self.form_type = normalize_form_type(self.form_type)


class PdfIndex:
    """Container that keeps every indexed PDF grouped by MB/form type."""

    def __init__(self, *, preferred_period: Optional[str] = None) -> None:
        self._entries: Dict[str, Dict[str, List[IndexedPdf]]] = defaultdict(
            lambda: {"bu": [], "bs": []}
        )
        self.unclassified_files: List[Path] = []
        self.report_notes: MutableMapping[str, MutableMapping[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.preferred_period = normalize_period_label(preferred_period)

    # -- container protocol -------------------------------------------------

    @property
    def entries(self) -> Dict[str, Dict[str, List[IndexedPdf]]]:
        return self._entries

    def __contains__(self, mb: str) -> bool:  # pragma: no cover - passthrough
        return normalize_mb(mb) in self._entries

    def __getitem__(self, mb: str) -> Dict[str, List[IndexedPdf]]:
        return self._entries[normalize_mb(mb)]

    # -- registration -------------------------------------------------------

    def register(self, entry: IndexedPdf) -> None:
        bucket = self._entries[entry.mb][entry.form_type]
        bucket.append(entry)
        bucket.sort(key=_entry_sort_key, reverse=True)
        if len(bucket) > 1:
            self._note_duplicate(entry.mb, entry.form_type, bucket)

    # -- retrieval ----------------------------------------------------------

    def get_latest(
        self,
        mb: str,
        form_type: str,
        *,
        preferred_period: Optional[str] = None,
    ) -> Optional[IndexedPdf]:
        form_type = normalize_form_type(form_type)
        preferred_period = normalize_period_label(
            preferred_period or self.preferred_period
        )
        entries = self._entries.get(normalize_mb(mb), {}).get(form_type, [])
        if not entries:
            return None
        if preferred_period is None:
            return entries[0]
        for entry in entries:
            if entry.period and entry.period.label == preferred_period:
                return entry
        return entries[0]

    # -- diagnostics --------------------------------------------------------

    def _note_duplicate(
        self, mb: str, form_type: str, entries: Iterable[IndexedPdf]
    ) -> None:
        paths = ", ".join(str(entry.path) for entry in entries)
        message = (
            "Multiple %s forms detected for MB %s. Selection is based on period"
            " and modification time. Candidates: %s"
        ) % (form_type.upper(), mb, paths)
        LOGGER.warning(message)
        self.report_notes[mb][form_type].append(message)


# ---------------------------------------------------------------------------
# Public helpers


def build_index(
    directory: Union[str, Path, None] = None,
    *,
    period_preference: Optional[str] = None,
    ocr_provider=ocr_engine,
    anchors_module=anchors,
    logger: Optional[logging.Logger] = None,
) -> PdfIndex:
    """Scan ``directory`` (defaults to ``Path.cwd()``) for PDF files.

    Parameters
    ----------
    directory:
        Directory whose tree should be indexed.  ``None`` defaults to the
        current working directory.
    period_preference:
        Desired period label (e.g. ``"current"`` or ``"previous"``).  The
        hint is used when retrieving entries with
        :func:`get_pdf_for`/``PdfIndex.get_latest``.
    ocr_provider:
        Module or object exposing OCR helpers.  It is expected to provide
        either ``get_or_run_ocr`` or ``extract_text``/``extract``.  Tests
        can inject lightweight stubs via this argument.
    anchors_module:
        Module exposing anchor detection helpers/patterns.  If specialised
        helpers are absent, the implementation falls back to spec-defined
        regular expressions.
    logger:
        Optional logger used for status messages.  Defaults to the module
        level logger.
    """

    logger = logger or LOGGER
    directory_path = Path(directory or Path.cwd())
    index = PdfIndex(preferred_period=period_preference)

    for pdf_path in sorted(directory_path.rglob("*.pdf")):
        try:
            ocr_payload = _run_ocr(pdf_path, ocr_provider)
        except Exception:  # pragma: no cover - exceptional path
            logger.exception("Failed to run OCR for %s", pdf_path)
            index.unclassified_files.append(pdf_path)
            continue

        text = _extract_text(ocr_payload)
        mb = _detect_maticni_broj(ocr_payload, text, anchors_module)
        if not mb:
            message = f"Unable to locate Matični broj inside {pdf_path}"
            logger.warning(message)
            index.unclassified_files.append(pdf_path)
            continue

        form_types = _detect_form_type(ocr_payload, text, anchors_module)
        if not form_types:
            message = f"Unable to determine form type for {pdf_path} (MB {mb})"
            logger.warning(message)
            index.report_notes[mb]["unknown"].append(message)
            continue

        period = _detect_period(ocr_payload, text, anchors_module)
        base_kwargs = dict(
            path=pdf_path,
            mb=mb,
            mtime=pdf_path.stat().st_mtime,
            period=period,
        )
        for form_type in form_types:
            entry = IndexedPdf(form_type=form_type, **base_kwargs)
            if period and period.raw:
                entry.notes.append(f"period:{period.raw}")
            index.register(entry)

    return index


def get_pdf_for(
    mb: str,
    form_type: str,
    *,
    index: Optional[PdfIndex] = None,
    period_preference: Optional[str] = None,
    default_directory: Union[str, Path, None] = None,
) -> Optional[IndexedPdf]:
    """Convenience wrapper returning the preferred PDF for ``mb``.

    Parameters
    ----------
    index:
        Optional existing :class:`PdfIndex`.  When not supplied the
        directory (``default_directory``) is indexed on-demand.
    period_preference:
        Override the period hint specified during :func:`build_index`.
    default_directory:
        Directory to scan if the index needs to be built lazily.
    """

    if index is None:
        index = build_index(default_directory, period_preference=period_preference)
    return index.get_latest(mb, form_type, preferred_period=period_preference)


# ---------------------------------------------------------------------------
# Internals


def normalize_mb(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    return digits.lstrip("0") or digits or value


def normalize_form_type(value: str) -> str:
    value = (value or "").strip().lower()
    if value in FORM_TYPES:
        return value
    if "uspeh" in value or value == "bu":
        return "bu"
    if "stanj" in value or value == "bs":
        return "bs"
    return value or "unknown"


def normalize_period_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip().lower()
    mapping = {
        "текућа": "current",
        "текuca": "current",
        "tekuca": "current",
        "current": "current",
        "pret": "previous",
        "прет": "previous",
        "previous": "previous",
    }
    return mapping.get(value, value or None)


def _entry_sort_key(entry: IndexedPdf) -> Tuple[int, float]:
    period_rank = 0
    if entry.period:
        period_rank = _period_rank(entry.period)
    return (period_rank, entry.mtime)


def _period_rank(period: PeriodInfo) -> int:
    label = normalize_period_label(period.label)
    year = period.year or 0
    if label == "current":
        base = 2
    elif label == "previous":
        base = 1
    else:
        base = 0
    return base * 10_000 + year


def _run_ocr(pdf_path: Path, provider) -> object:
    if provider is None:
        raise RuntimeError("OCR provider is not configured")

    candidates = [
        getattr(provider, "get_or_run_ocr", None),
        getattr(provider, "extract", None),
        getattr(provider, "extract_text", None),
        getattr(provider, "process", None),
    ]
    for candidate in candidates:
        if callable(candidate):
            return candidate(pdf_path)

    raise RuntimeError(
        "ocr_engine does not expose a supported API. Expected one of"
        " get_or_run_ocr(), extract(), extract_text() or process()."
    )


def _extract_text(ocr_payload: object) -> str:
    if ocr_payload is None:
        return ""
    if isinstance(ocr_payload, str):
        return ocr_payload
    if isinstance(ocr_payload, dict):
        for key in ("text", "plain_text", "content"):
            value = ocr_payload.get(key)
            if isinstance(value, str):
                return value
    text = getattr(ocr_payload, "text", None)
    if isinstance(text, str):
        return text
    return str(ocr_payload)


def _detect_maticni_broj(ocr_payload: object, text: str, anchors_module) -> Optional[str]:
    detectors = [
        getattr(anchors_module, "extract_maticni_broj", None),
        getattr(anchors_module, "detect_maticni_broj", None),
        getattr(anchors_module, "find_maticni_broj", None),
    ]
    for detector in detectors:
        if not callable(detector):
            continue
        result = detector(ocr_payload)
        if isinstance(result, (list, tuple)):
            result = result[0] if result else None
        if result:
            return normalize_mb(str(result))

    pattern = getattr(anchors_module, "MB_REGEX", None)
    if isinstance(pattern, str):
        pattern = re.compile(pattern, re.IGNORECASE)
    if pattern is None:
        pattern = re.compile(r"(?<!\d)(\d{8,9})(?!\d)")

    match = pattern.search(text)
    if match:
        return normalize_mb(match.group(1))
    return None


def _detect_form_type(ocr_payload: object, text: str, anchors_module) -> List[str]:
    detectors = [
        getattr(anchors_module, "detect_form_type", None),
        getattr(anchors_module, "classify_form", None),
        getattr(anchors_module, "detect_form", None),
    ]
    for detector in detectors:
        if not callable(detector):
            continue
        result = detector(ocr_payload)
        forms = _coerce_form_types(result)
        if forms:
            return forms

    form_patterns = getattr(anchors_module, "FORM_HEADER_PATTERNS", None)
    if not isinstance(form_patterns, dict):
        form_patterns = {}
    pattern_bu = _ensure_regex(
        form_patterns.get("bu"), r"(?i)билан[сc]\s+успеха|bilans\s+uspeha"
    )
    pattern_bs = _ensure_regex(
        form_patterns.get("bs"), r"(?i)билан[сc]\s+стања|bilans\s+stanja"
    )

    detected: List[str] = []
    if pattern_bu.search(text):
        detected.append("bu")
    if pattern_bs.search(text):
        detected.append("bs")
    return detected


def _coerce_form_types(candidate) -> List[str]:
    forms: List[str] = []

    def _append(value) -> None:
        if value is None:
            return
        normalized = normalize_form_type(str(value))
        if normalized in FORM_TYPES and normalized not in forms:
            forms.append(normalized)

    if candidate is None:
        return forms

    if isinstance(candidate, dict):
        if "form_types" in candidate:
            return _coerce_form_types(candidate["form_types"])
        if "form_type" in candidate:
            _append(candidate["form_type"])
            return forms
        if "type" in candidate:
            _append(candidate["type"])
            return forms

    if isinstance(candidate, (list, tuple, set)):
        for item in candidate:
            for value in _coerce_form_types(item):
                if value not in forms:
                    forms.append(value)
        return forms

    _append(candidate)
    return forms


def _detect_period(
    ocr_payload: object, text: str, anchors_module
) -> Optional[PeriodInfo]:
    detectors = [
        getattr(anchors_module, "detect_period", None),
        getattr(anchors_module, "detect_periods", None),
    ]
    for detector in detectors:
        if not callable(detector):
            continue
        result = detector(ocr_payload)
        period = _coerce_period(result)
        if period:
            return period

    # Fallback: look for a four digit year and mark it as generic period.
    years = [int(year) for year in re.findall(r"(20\d{2})", text or "")]
    if years:
        return PeriodInfo(year=max(years))
    return None


def _coerce_period(candidate) -> Optional[PeriodInfo]:
    if candidate is None:
        return None
    if isinstance(candidate, PeriodInfo):
        return candidate
    if isinstance(candidate, dict):
        label = candidate.get("label") or candidate.get("period")
        year = candidate.get("year")
        if isinstance(year, str) and year.isdigit():
            year = int(year)
        if isinstance(year, (int, float)):
            year = int(year)
        return PeriodInfo(label=label, year=year, raw=candidate)
    if isinstance(candidate, (list, tuple)):
        for item in candidate:
            period = _coerce_period(item)
            if period:
                return period
        return None
    if isinstance(candidate, str):
        match = re.search(r"(20\d{2})", candidate)
        year = int(match.group(1)) if match else None
        return PeriodInfo(label=candidate, year=year, raw=candidate)
    return PeriodInfo(raw=candidate)


def _ensure_regex(pattern, fallback: str) -> re.Pattern:
    if isinstance(pattern, re.Pattern):
        return pattern
    if isinstance(pattern, str):
        return re.compile(pattern, re.IGNORECASE)
    return re.compile(fallback)


__all__ = [
    "IndexedPdf",
    "PdfIndex",
    "PeriodInfo",
    "build_index",
    "get_pdf_for",
]
