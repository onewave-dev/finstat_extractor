"""Shared helpers for extracting table values from OCR TSV output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ocr import anchors
from ocr.ocr_engine import OcrPage, OcrResult

from .models import ExtractionMessage, ExtractionResult
from .numeric import NumericParseError, normalize_numeric_string


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

    anchor_map = anchors.build_anchor_map()
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

    for page in ocr_result.pages:
        page_lines = list(_build_lines(page))
        if not page_lines:
            continue

        page_columns = _detect_year_columns(page_lines)
        for line in _find_anchor_lines(page_lines, anchor_def):
            detected_anchor = True
            last_anchor_line = line
            column, column_label, column_diag = _select_column(
                page_columns, year_preference
            )
            if column:
                last_column = column

            cluster, clusters = _locate_numeric_cluster(line, column)
            if cluster is None:
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
                            "available_columns": sorted(page_columns.keys()),
                            "candidate_values": [c.text for c in clusters],
                        },
                    )
                )
                continue

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
    for line in lines:
        text = line.text.strip()
        lowered = text.lower()
        if anchor_def.pattern.search(lowered) or any(
            syn in lowered for syn in lowered_synonyms
        ):
            yield line


def _detect_year_columns(lines: Sequence[OcrLine]) -> Dict[str, ColumnPosition]:
    result: Dict[str, ColumnPosition] = {}
    reference_year = anchors.get_reference_year()
    synonyms = anchors.get_year_column_synonyms(reference_year=reference_year)
    numeric_labels = {
        reference_year: "current",
        reference_year - 1: "previous",
    }

    for line in lines:
        norm_words = [(word, _normalise_token(word.text)) for word in line.words]
        filtered_words = [(w, tok) for w, tok in norm_words if tok]
        tokens = [tok for _, tok in filtered_words]

        for label, variants in synonyms.items():
            for variant in variants:
                variant_tokens = [_normalise_token(part) for part in variant.split()]
                variant_tokens = [tok for tok in variant_tokens if tok]
                if not variant_tokens:
                    continue
                match = _locate_token_sequence(filtered_words, variant_tokens)
                if match is None:
                    continue
                words = match
                left = min(word.left for word in words)
                right = max(word.right for word in words)
                top = min(word.top for word in words)
                bottom = max(word.bottom for word in words)
                candidate = ColumnPosition(
                    label=label,
                    page_number=line.page_number,
                    left=left,
                    right=right,
                    top=top,
                    bottom=bottom,
                    text=line.text.strip(),
                )
                existing = result.get(label)
                if existing is None or candidate.top < existing.top:
                    result[label] = candidate
                break

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
            existing = result.get(label)
            if existing is None or candidate.top < existing.top:
                result[label] = candidate

    return result


def _normalise_token(token: str) -> str:
    return "".join(ch for ch in token.lower() if ch.isalnum())


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
    line: OcrLine, column: Optional[ColumnPosition]
) -> Tuple[Optional[NumericCluster], List[NumericCluster]]:
    clusters = _build_numeric_clusters(line)
    if not clusters:
        return None, clusters

    anchor_right = line.right
    filtered = [cluster for cluster in clusters if cluster.right >= anchor_right - 5]
    if not filtered:
        filtered = clusters

    if column:
        column_center = column.center_x
        filtered.sort(
            key=lambda cluster: (
                abs(cluster.center_x - column_center),
                abs(cluster.left - column_center),
            )
        )
    else:
        filtered.sort(
            key=lambda cluster: (
                0 if cluster.left >= anchor_right else 1,
                abs(cluster.left - anchor_right),
            )
        )

    return filtered[0], clusters


def _build_numeric_clusters(line: OcrLine) -> List[NumericCluster]:
    clusters: List[NumericCluster] = []
    current: List[OcrWord] = []

    for word in line.words:
        if _is_numeric_like(word.text):
            current.append(word)
            continue
        if current:
            clusters.append(_make_cluster(current))
            current = []

    if current:
        clusters.append(_make_cluster(current))

    return clusters


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


__all__ = ["extract_field_from_ocr"]

