"""Shared helpers for extracting table values from OCR TSV output."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ocr import anchors
from ocr.ocr_engine import OcrPage, OcrResult

from .models import ExtractionMessage, ExtractionResult
from .numeric import NumericParseError, normalize_numeric_string


_VISUALLY_SIMILAR_TRANSLATION = str.maketrans(
    {
        "A": "А",
        "B": "В",
        "C": "С",
        "E": "Е",
        "H": "Н",
        "h": "н",
        "K": "К",
        "k": "к",
        "M": "М",
        "m": "м",
        "O": "О",
        "P": "Р",
        "T": "Т",
        "t": "т",
        "X": "Х",
        "Y": "У",
        "a": "а",
        "c": "с",
        "e": "е",
        "o": "о",
        "p": "р",
        "x": "х",
        "y": "у",
        "J": "Ј",
        "j": "ј",
    }
)


def _normalise_mixed_script(text: str) -> str:
    """Return ``text`` where visually identical Latin characters use Cyrillic forms."""

    return text.translate(_VISUALLY_SIMILAR_TRANSLATION)


@dataclass
class OcrWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    raw: Dict[str, str]

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def normalized_text(self) -> str:
        return self.text.strip()


@dataclass
class OcrLine:
    page_number: int
    block_num: int
    par_num: int
    line_num: int
    words: List[OcrWord]

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words if word.text)

    @property
    def left(self) -> int:
        return min(word.left for word in self.words)

    @property
    def right(self) -> int:
        return max(word.right for word in self.words)

    @property
    def top(self) -> int:
        return min(word.top for word in self.words)

    @property
    def bottom(self) -> int:
        return max(word.bottom for word in self.words)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


@dataclass
class ColumnPosition:
    label: str
    page_number: int
    left: int
    right: int
    top: int
    bottom: int
    text: str

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0


@dataclass
class NumericCluster:
    text: str
    left: int
    right: int
    top: int
    bottom: int
    words: List[OcrWord]

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0


def _infer_reference_year(ocr_result: OcrResult) -> int:
    pattern = anchors.YEAR_FOUR_DIGIT_PATTERN
    candidate_years = set()

    def _collect(text: Optional[str]) -> None:
        if not text:
            return
        for match in pattern.finditer(text):
            try:
                year_value = int(match.group("year"))
            except (TypeError, ValueError):
                continue
            if year_value >= 2000:
                candidate_years.add(year_value)

    _collect(ocr_result.text)
    for page in ocr_result.pages:
        lines = page.text.splitlines()
        for line in lines[:5]:
            _collect(line)

    if candidate_years:
        return max(candidate_years)
    return date.today().year


def extract_field_from_ocr(
    ocr_result: OcrResult,
    *,
    anchor_key: str,
    field_name: str,
    year_preference: Optional[str] = "current",
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> ExtractionResult:
    result = ExtractionResult(field_name=field_name)
    if not ocr_result.pages:
        result.add_error("no_pages", "OCR result does not contain any pages")
        return result

    reference_year = _infer_reference_year(ocr_result)
    anchor_map = anchors.build_anchor_map(reference_year=reference_year)
    anchor_def = anchor_map.get(anchor_key)
    if anchor_def is None:
        result.add_error(
            "anchor_definition_missing",
            "Anchor definition is not available",
            anchor_key=anchor_key,
        )
        return result

    detected_anchor = False
    collected_errors: List[ExtractionMessage] = []
    last_anchor_line: Optional[OcrLine] = None
    last_column: Optional[ColumnPosition] = None
    last_columns: Optional[Dict[str, ColumnPosition]] = None
    last_aop_column: Optional[ColumnPosition] = None

    for page in ocr_result.pages:
        page_lines = list(_build_lines(page))
        if not page_lines:
            continue

        page_columns = _detect_year_columns(page_lines, reference_year=reference_year)
        if page_columns:
            last_columns = page_columns

        page_aop_column = _detect_aop_column(page_lines)
        if page_aop_column is not None:
            last_aop_column = page_aop_column

        candidate_columns: Dict[str, ColumnPosition] = {}
        if page_columns:
            candidate_columns = page_columns
        elif last_columns:
            candidate_columns = last_columns

        aop_column = page_aop_column or last_aop_column

        for line in _find_anchor_lines(page_lines, anchor_def):
            detected_anchor = True
            last_anchor_line = line
            column, column_label, column_diag = _select_column(
                candidate_columns, year_preference
            )
            if column:
                last_column = column

            candidate_clusters, clusters = _locate_numeric_cluster(
                line, column, page_lines, aop_column, candidate_columns
            )
            if not candidate_clusters:
                if column_diag and column is None:
                    collected_errors.append(column_diag)
                collected_errors.append(
                    ExtractionMessage(
                        code="value_not_found",
                        message="No numeric value detected next to the anchor",
                        context={
                            "page": page.page_number,
                            "anchor_text": line.text.strip(),
                                "anchor_bbox": line.bbox,
                                "available_columns": sorted(candidate_columns.keys()),
                                "candidate_values": [c.text for c in clusters],
                            },
                        )
                    )
                continue

            for cluster in candidate_clusters:
                try:
                    parse_result = normalize_numeric_string(
                        cluster.text,
                        min_value=min_value,
                        max_value=max_value,
                    )
                except NumericParseError as exc:
                    if column_diag and column is None:
                        collected_errors.append(column_diag)
                    collected_errors.append(
                        ExtractionMessage(
                            code="value_normalization_failed",
                            message=str(exc),
                            context={
                                "page": page.page_number,
                                "raw_text": cluster.text,
                                "anchor_text": line.text.strip(),
                            },
                        )
                    )
                    continue

                result.value = parse_result.value
                result.raw_text = cluster.text
                result.normalized_text = parse_result.normalized_text
                result.page_number = page.page_number
                result.anchor_text = line.text.strip()
                result.anchor_bbox = line.bbox
                if column:
                    result.column_label = column_label
                    result.column_bbox = column.bbox
                if column_diag:
                    if column is None:
                        result.warnings.append(column_diag)
                    else:
                        result.warnings.append(column_diag)
                return result

    if detected_anchor:
        if last_anchor_line is not None and result.anchor_text is None:
            result.page_number = last_anchor_line.page_number
            result.anchor_text = last_anchor_line.text.strip()
            result.anchor_bbox = last_anchor_line.bbox
        if last_column is not None and result.column_label is None:
            result.column_label = last_column.label
            result.column_bbox = last_column.bbox
        result.errors.extend(collected_errors)
        if not result.errors:
            result.add_error(
                "value_not_found",
                "Anchor detected but numeric value could not be resolved",
                anchor_key=anchor_key,
            )
    else:
        result.add_error(
            "anchor_not_found",
            "Anchor text was not located in the OCR output",
            anchor_key=anchor_key,
        )

    return result


def _build_lines(page: OcrPage) -> Iterable[OcrLine]:
    buckets: Dict[Tuple[int, int, int], List[OcrWord]] = {}
    for row in page.tsv:
        text = row.get("text", "").strip()
        if not text:
            continue
        try:
            level = int(row.get("level", 0))
        except ValueError:
            continue
        if level != 5:
            continue
        try:
            block_num = int(row.get("block_num", 0))
            par_num = int(row.get("par_num", 0))
            line_num = int(row.get("line_num", 0))
            left = int(float(row.get("left", 0)))
            top = int(float(row.get("top", 0)))
            width = int(float(row.get("width", 0)))
            height = int(float(row.get("height", 0)))
        except (TypeError, ValueError):
            continue

        key = (block_num, par_num, line_num)
        word = OcrWord(
            text=row.get("text", ""),
            left=left,
            top=top,
            width=width,
            height=height,
            raw=row,
        )
        buckets.setdefault(key, []).append(word)

    for (block_num, par_num, line_num), words in buckets.items():
        if not words:
            continue
        words.sort(key=lambda w: w.left)
        yield OcrLine(
            page_number=page.page_number,
            block_num=block_num,
            par_num=par_num,
            line_num=line_num,
            words=words,
        )


def _find_anchor_lines(
    lines: Sequence[OcrLine], anchor_def: anchors.AnchorDefinition
) -> Iterable[OcrLine]:
    lowered_synonyms = [syn.lower() for syn in anchor_def.synonyms]
    normalized_synonyms = [
        _normalise_mixed_script(syn).lower() for syn in anchor_def.synonyms
    ]
    matched_line_ids: Set[int] = set()
    for line in lines:
        text = line.text.strip()
        lowered = text.lower()
        normalized = _normalise_mixed_script(text)
        normalized_lower = normalized.lower()
        if (
            anchor_def.pattern.search(normalized)
            or anchor_def.pattern.search(text)
            or any(syn in lowered for syn in lowered_synonyms)
            or any(syn in normalized_lower for syn in normalized_synonyms)
        ):
            matched_line_ids.add(id(line))
            yield line

    band_pattern = anchors.ROW_ANCHOR_PATTERNS.get(anchor_def.key)
    if band_pattern is None:
        return

    for band in _group_words_into_vertical_bands(lines):
        band_text = band.text
        if not band_text:
            continue
        normalized_band_text = _normalise_mixed_script(band_text)
        if not band_pattern.search(normalized_band_text):
            continue
        entries = band._iter_entries()
        if not entries:
            continue

        seen_line_ids: Set[int] = set()
        lines_in_band: List[OcrLine] = []
        for line, _ in entries:
            line_id = id(line)
            if line_id in seen_line_ids:
                continue
            seen_line_ids.add(line_id)
            lines_in_band.append(line)

        if any(id(line) in matched_line_ids for line in lines_in_band):
            continue

        seen_word_ids: Set[int] = set()
        words: List[OcrWord] = []
        for line in lines_in_band:
            for word in line.words:
                if not word.text:
                    continue
                word_id = id(word)
                if word_id in seen_word_ids:
                    continue
                seen_word_ids.add(word_id)
                words.append(word)

        if not words:
            continue

        words.sort(key=lambda word: (word.top, word.left))
        reference_line = min(
            lines_in_band, key=lambda item: (item.top, item.left)
        )

        yield OcrLine(
            page_number=reference_line.page_number,
            block_num=reference_line.block_num,
            par_num=reference_line.par_num,
            line_num=reference_line.line_num,
            words=words,
        )


def _detect_year_columns(
    lines: Sequence[OcrLine], *, reference_year: Optional[int] = None
) -> Dict[str, ColumnPosition]:
    result: Dict[str, ColumnPosition] = {}
    resolved_reference_year = anchors.get_reference_year(reference_year)
    synonyms = anchors.get_year_column_synonyms(
        reference_year=resolved_reference_year
    )
    numeric_labels = {
        resolved_reference_year: "current",
        resolved_reference_year - 1: "previous",
    }

    bands = _group_words_into_vertical_bands(lines)

    for label, variants in synonyms.items():
        best_match: Optional[ColumnPosition] = None
        for band in bands:
            filtered_words = band.filtered_words
            if not filtered_words:
                continue
            for variant in variants:
                variant_tokens = [_normalise_token(part) for part in variant.split()]
                variant_tokens = [tok for tok in variant_tokens if tok]
                if not variant_tokens:
                    continue
                match = _locate_token_sequence(filtered_words, variant_tokens)
                if match is None:
                    continue
                words = match
                candidate = ColumnPosition(
                    label=label,
                    page_number=band.page_number,
                    left=min(word.left for word in words),
                    right=max(word.right for word in words),
                    top=min(word.top for word in words),
                    bottom=max(word.bottom for word in words),
                    text=band.text.strip(),
                )
                best_match = _pick_topmost(best_match, candidate)
                break
        if best_match is not None:
            result[label] = best_match
            continue

        pattern = anchors.YEAR_COLUMN_PATTERNS.get(label)
        if pattern is None:
            continue
        for band in bands:
            band_text = band.text.strip()
            if not band_text:
                continue
            if not pattern.search(band_text):
                continue
            candidate = ColumnPosition(
                label=label,
                page_number=band.page_number,
                left=band.left,
                right=band.right,
                top=band.top,
                bottom=band.bottom,
                text=band_text,
            )
            result[label] = _pick_topmost(result.get(label), candidate)

    for line in lines:
        for word in line.words:
            match = anchors.YEAR_FOUR_DIGIT_PATTERN.search(word.text)
            if not match:
                continue
            try:
                year_value = int(match.group("year"))
            except (TypeError, ValueError):
                continue
            label = numeric_labels.get(year_value)
            if label is None:
                continue
            candidate = ColumnPosition(
                label=label,
                page_number=line.page_number,
                left=word.left,
                right=word.right,
                top=word.top,
                bottom=word.bottom,
                text=line.text.strip(),
            )
            result[label] = _pick_topmost(result.get(label), candidate)

    return result


def _detect_aop_column(lines: Sequence[OcrLine]) -> Optional[ColumnPosition]:
    target_tokens = {"aop", "аоп", "аор"}
    best: Optional[ColumnPosition] = None

    for band in _group_words_into_vertical_bands(lines):
        tokens = [token for _, token in band.filtered_words]
        if not tokens:
            continue
        expanded_tokens = set(tokens)

        def _add_run_sequences(run_tokens: Sequence[str]) -> None:
            if len(run_tokens) < 2:
                return
            for length in range(2, len(run_tokens) + 1):
                for offset in range(len(run_tokens) - length + 1):
                    expanded_tokens.add("".join(run_tokens[offset : offset + length]))

        # Some OCR outputs render the "AOP" column header as three stacked
        # single-letter tokens ("А", "О", "П"). Collapse such runs so that
        # they can be matched against the target tokens.
        run_start: Optional[int] = None
        for index, token in enumerate(tokens):
            if len(token) == 1:
                if run_start is None:
                    run_start = index
                continue
            if run_start is not None:
                _add_run_sequences(tokens[run_start:index])
                run_start = None
        if run_start is not None:
            _add_run_sequences(tokens[run_start:])

        if not any(token in target_tokens for token in expanded_tokens):
            continue
        candidate = ColumnPosition(
            label="aop",
            page_number=band.page_number,
            left=band.left,
            right=band.right,
            top=band.top,
            bottom=band.bottom,
            text=band.text.strip(),
        )
        best = _pick_topmost(best, candidate)

    return best


def _normalise_token(token: str) -> str:
    normalized = _normalise_mixed_script(token)
    return "".join(ch for ch in normalized.lower() if ch.isalnum())


def _locate_token_sequence(
    words_with_tokens: Sequence[Tuple[OcrWord, str]],
    target_tokens: Sequence[str],
) -> Optional[List[OcrWord]]:
    token_values = [token for _, token in words_with_tokens]
    length = len(target_tokens)
    for start in range(len(token_values) - length + 1):
        window = token_values[start : start + length]
        if window == list(target_tokens):
            return [words_with_tokens[idx][0] for idx in range(start, start + length)]
    return None


class _ColumnBand:
    __slots__ = ("page_number", "left", "right", "top", "bottom", "_entries")

    def __init__(self, line: OcrLine, word: OcrWord) -> None:
        self.page_number = line.page_number
        self.left = word.left
        self.right = word.right
        self.top = word.top
        self.bottom = word.bottom
        self._entries: List[Tuple[OcrLine, OcrWord]] = [(line, word)]

    def add(self, line: OcrLine, word: OcrWord) -> None:
        self._entries.append((line, word))
        self.left = min(self.left, word.left)
        self.right = max(self.right, word.right)
        self.top = min(self.top, word.top)
        self.bottom = max(self.bottom, word.bottom)

    @property
    def filtered_words(self) -> List[Tuple[OcrWord, str]]:
        result: List[Tuple[OcrWord, str]] = []
        for _, word in self._iter_entries():
            token = _normalise_token(word.text)
            if token:
                result.append((word, token))
        return result

    @property
    def text(self) -> str:
        lines_map: Dict[Tuple[int, int, int, int], List[OcrWord]] = {}
        for line, word in self._entries:
            key = (line.page_number, line.block_num, line.par_num, line.line_num)
            lines_map.setdefault(key, []).append(word)

        parts: List[str] = []
        for words in sorted(
            lines_map.values(),
            key=lambda words: (
                min(word.top for word in words),
                min(word.left for word in words),
            ),
        ):
            sorted_words = sorted(words, key=lambda word: word.left)
            line_text = " ".join(word.text for word in sorted_words if word.text.strip())
            if not line_text:
                continue
            if parts:
                parts.append("\n")
            parts.append(line_text)
        return "".join(parts)

    def _iter_entries(self) -> List[Tuple[OcrLine, OcrWord]]:
        return sorted(self._entries, key=lambda item: (item[1].top, item[1].left))


def _group_words_into_vertical_bands(lines: Sequence[OcrLine]) -> List[_ColumnBand]:
    bands: List[_ColumnBand] = []
    for line in lines:
        for word in line.words:
            if not word.text.strip():
                continue
            band = _find_matching_band(bands, word)
            if band is None:
                bands.append(_ColumnBand(line, word))
            else:
                band.add(line, word)
    bands.sort(key=lambda band: band.left)
    return bands


def _find_matching_band(bands: Sequence[_ColumnBand], word: OcrWord) -> Optional[_ColumnBand]:
    for band in bands:
        if _horizontal_overlap(word.left, word.right, band.left, band.right):
            return band
    return None


def _horizontal_overlap(left1: int, right1: int, left2: int, right2: int) -> bool:
    word_width = max(right1 - left1, 1)
    band_width = max(right2 - left2, 1)
    tolerance = max(12, min(max(word_width, band_width), 60))
    return not (right1 < left2 - tolerance or right2 < left1 - tolerance)


def _pick_topmost(
    existing: Optional[ColumnPosition], candidate: Optional[ColumnPosition]
) -> Optional[ColumnPosition]:
    if candidate is None:
        return existing
    if existing is None or candidate.top < existing.top:
        return candidate
    return existing


def _select_column(
    columns: Dict[str, ColumnPosition],
    year_preference: Optional[str],
) -> Tuple[Optional[ColumnPosition], Optional[str], Optional[ExtractionMessage]]:
    available = set(columns.keys())
    preferred = year_preference.lower() if year_preference else None
    column_diag: Optional[ExtractionMessage] = None

    if preferred and preferred in columns:
        return columns[preferred], preferred, None

    for fallback in ("current", "previous"):
        if fallback == preferred:
            continue
        if fallback in columns:
            column_diag = ExtractionMessage(
                code="column_fallback_used",
                message="Preferred year column missing; using fallback column",
                context={
                    "preferred": preferred,
                    "selected": fallback,
                    "available_columns": sorted(available),
                },
            )
            return columns[fallback], fallback, column_diag

    column_diag = ExtractionMessage(
        code="column_not_found",
        message="Year columns could not be detected on the anchor page",
        context={
            "preferred": preferred,
            "available_columns": sorted(available),
        },
    )
    return None, None, column_diag


def _locate_numeric_cluster(
    anchor_line: OcrLine,
    column: Optional[ColumnPosition],
    page_lines: Sequence[OcrLine],
    aop_column: Optional[ColumnPosition],
    columns: Optional[Dict[str, ColumnPosition]] = None,
) -> Tuple[List[NumericCluster], List[NumericCluster]]:
    anchor_line_right = anchor_line.right
    anchor_top = anchor_line.top
    anchor_bottom = anchor_line.bottom

    anchor_text_right_candidates = [
        word.right
        for word in anchor_line.words
        if word.text.strip() and not _is_numeric_like(word.text)
    ]
    anchor_right = max(anchor_text_right_candidates, default=anchor_line_right)

    relevant_words: List[OcrWord] = []
    all_relevant_words: List[OcrWord] = []

    column_tolerance = 8
    column_left_bound: Optional[float] = None
    column_right_bound: Optional[float] = None
    if column is not None:
        column_left_bound, column_right_bound = _compute_column_bounds(
            column,
            columns,
            anchor_right=anchor_right,
            anchor_line_right=anchor_line_right,
        )

    def _word_in_selected_column(word: OcrWord) -> bool:
        if column is None or column_left_bound is None or column_right_bound is None:
            return True
        column_left = column_left_bound - column_tolerance
        column_right = column_right_bound + column_tolerance
        word_center = (word.left + word.right) / 2.0
        if word.right < column_left or word.left > column_right:
            return False
        if column_left <= word_center <= column_right:
            return True
        if word.left >= column_left and word.right <= column_right:
            return True
        return False

    for word in anchor_line.words:
        if not word.text.strip():
            continue
        all_relevant_words.append(word)
        if not _word_in_selected_column(word):
            continue
        relevant_words.append(word)

    for line in page_lines:
        if line is anchor_line:
            continue
        if line.page_number != anchor_line.page_number:
            continue
        if line.bottom < anchor_top or line.top > anchor_bottom:
            continue
        if line.left < anchor_right - 5:
            continue
        for word in line.words:
            if not word.text.strip():
                continue
            if word.right < anchor_right - 5:
                continue
            all_relevant_words.append(word)
            if not _word_in_selected_column(word):
                continue
            relevant_words.append(word)

    clusters = _build_numeric_clusters(relevant_words)
    used_column_words = bool(relevant_words)
    if column is not None and not clusters:
        clusters = _build_numeric_clusters(all_relevant_words)
        used_column_words = False
    if not clusters:
        return [], clusters

    if (
        column is not None
        and used_column_words
        and column_left_bound is not None
        and column_right_bound is not None
    ):
        column_left = column_left_bound - column_tolerance
        column_right = column_right_bound + column_tolerance
        column_limited_clusters: List[NumericCluster] = []
        for cluster in clusters:
            words_in_column = [
                word for word in cluster.words if _word_in_selected_column(word)
            ]
            if not words_in_column:
                continue
            for sub_cluster in _build_numeric_clusters(words_in_column):
                column_limited_clusters.append(sub_cluster)
        clusters = column_limited_clusters
        if not clusters:
            return [], clusters

    filtered = [cluster for cluster in clusters if cluster.right >= anchor_right - 5]
    if not filtered:
        filtered = list(clusters)

    if aop_column is not None:
        filtered_no_aop = [
            cluster
            for cluster in filtered
            if not _cluster_in_aop_zone(cluster, aop_column)
        ]
        if not filtered_no_aop:
            filtered_no_aop = [
                cluster
                for cluster in clusters
                if not _cluster_in_aop_zone(cluster, aop_column)
            ]
        if filtered_no_aop:
            filtered = filtered_no_aop

    anchor_sort_right = anchor_line_right

    def _distance_key(cluster: NumericCluster) -> Tuple[float, float, float]:
        if column:
            column_center = column.center_x
            return (
                abs(cluster.center_x - column_center),
                abs(cluster.left - column_center),
                cluster.left,
            )
        return (
            0.0 if cluster.left >= anchor_sort_right else 1.0,
            abs(cluster.left - anchor_sort_right),
            cluster.left,
        )

    high_priority: List[NumericCluster] = []
    low_priority: List[NumericCluster] = []
    for cluster in filtered:
        if _looks_like_formula(cluster.text):
            low_priority.append(cluster)
        else:
            high_priority.append(cluster)

    high_priority.sort(key=_distance_key)
    low_priority.sort(key=_distance_key)

    ordered = high_priority + low_priority
    return ordered, clusters


def _build_numeric_clusters(words: Sequence[OcrWord]) -> List[NumericCluster]:
    clusters: List[NumericCluster] = []
    current: List[OcrWord] = []

    sorted_words = sorted(words, key=lambda word: (word.top, word.left))

    for word in sorted_words:
        if _is_numeric_like(word.text):
            if current:
                prev = current[-1]
                if _should_break_cluster(prev, word):
                    clusters.append(_make_cluster(current))
                    current = []
            current.append(word)
            continue
        if current:
            clusters.append(_make_cluster(current))
            current = []

    if current:
        clusters.append(_make_cluster(current))

    return clusters


def _should_break_cluster(prev: OcrWord, current: OcrWord) -> bool:
    horizontal_gap = current.left - prev.right
    if horizontal_gap > _cluster_gap_threshold(prev, current):
        return True

    vertical_gap = abs(current.top - prev.top)
    if vertical_gap > max(prev.height, current.height):
        return True

    return False


def _cluster_gap_threshold(prev: OcrWord, current: OcrWord) -> int:
    max_height = max(prev.height, current.height, 1)
    return max(15, int(max_height * 1.5))


def _is_numeric_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if any(ch.isdigit() for ch in stripped):
        return True
    return any(ch in "-−()+" for ch in stripped)


def _make_cluster(words: List[OcrWord]) -> NumericCluster:
    text = " ".join(word.text.strip() for word in words if word.text.strip())
    left = min(word.left for word in words)
    right = max(word.right for word in words)
    top = min(word.top for word in words)
    bottom = max(word.bottom for word in words)
    return NumericCluster(text=text, left=left, right=right, top=top, bottom=bottom, words=list(words))


def _looks_like_formula(text: str) -> bool:
    normalized = text.upper()
    return "+" in text or "=" in text or "АОП" in normalized


def _compute_column_bounds(
    column: ColumnPosition,
    columns: Optional[Dict[str, ColumnPosition]],
    *,
    anchor_right: Optional[int] = None,
    anchor_line_right: Optional[int] = None,
) -> Tuple[float, float]:
    same_page_columns: List[ColumnPosition] = []
    if columns:
        for candidate in columns.values():
            if candidate.page_number != column.page_number:
                continue
            same_page_columns.append(candidate)
    if not any(candidate is column for candidate in same_page_columns):
        same_page_columns.append(column)

    column_center = column.center_x
    left_neighbor: Optional[float] = None
    right_neighbor: Optional[float] = None
    for candidate in same_page_columns:
        center = candidate.center_x
        if center < column_center:
            if left_neighbor is None or center > left_neighbor:
                left_neighbor = center
        elif center > column_center:
            if right_neighbor is None or center < right_neighbor:
                right_neighbor = center

    width = max(column.right - column.left, 1)
    base_extension = max(width * 1.5, 80.0)

    if left_neighbor is not None:
        left_bound = (left_neighbor + column_center) / 2.0
    else:
        left_bound = column.left - base_extension
    if anchor_right is not None:
        left_bound = max(left_bound, float(anchor_right))

    if right_neighbor is not None:
        right_bound = (right_neighbor + column_center) / 2.0
    else:
        right_bound = column.right + base_extension
        if anchor_line_right is not None:
            right_bound = max(right_bound, float(anchor_line_right))

    if right_bound <= left_bound:
        right_bound = left_bound + max(width, base_extension / 2.0)

    return left_bound, right_bound


def _cluster_in_aop_zone(
    cluster: NumericCluster, aop_column: ColumnPosition
) -> bool:
    center = cluster.center_x
    margin = 3
    return (aop_column.left - margin) <= center <= (aop_column.right + margin)


__all__ = ["extract_field_from_ocr"]
