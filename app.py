"""Command line interface orchestrating the FinStat extraction workflow."""

from __future__ import annotations

import argparse
import csv
import inspect
import logging
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from io import excel_io
from index import pdf_index
from ocr import ocr_engine
from extract.models import ExtractionMessage, ExtractionResult

try:  # pragma: no cover - optional during tests
    from extract import bu as extract_bu  # type: ignore
except ImportError:  # pragma: no cover - optional during tests
    extract_bu = None  # type: ignore

try:  # pragma: no cover - optional during tests
    from extract import bs as extract_bs  # type: ignore
except ImportError:  # pragma: no cover - optional during tests
    extract_bs = None  # type: ignore


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_LOG_PATH = Path("logs") / "finstat_extractor.log"
REPORT_PATH = Path("report_missing.csv")

EXTRACTOR_NAMES: Dict[str, Dict[str, Sequence[str]]] = {
    "bu": {
        "revenue": ("extract_poslovni_prihodi", "extract_revenue"),
    },
    "bs": {
        "assets": (
            "extract_ukupna_aktiva",
            "extract_total_assets",
            "extract_assets",
        ),
        "capital_loss": (
            "extract_gubitak_iznad_visine_kapitala",
            "extract_capital_loss",
        ),
    },
}


@dataclass
class ProcessingStats:
    total_rows: int = 0
    rows_with_mb: int = 0
    rows_with_updates: int = 0
    values_written: int = 0
    values_skipped_existing: int = 0
    missing_fields: Counter[str] = field(default_factory=Counter)


@dataclass
class ReportEntry:
    mb: str
    missing_bu: bool = False
    missing_bs: bool = False
    missing_fields: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_csv_row(self) -> List[str]:
        return [
            self.mb,
            "yes" if self.missing_bu else "no",
            "yes" if self.missing_bs else "no",
            ",".join(sorted(set(self.missing_fields))),
            " | ".join(self.notes),
        ]


class DependencyError(RuntimeError):
    """Raised when external dependencies (Tesseract/Poppler) are missing."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract financial figures from PDFs and populate Excel columns G/H/I.",
    )
    parser.add_argument("--excel", required=True, help="Path to the Excel workbook that should be updated.")
    parser.add_argument(
        "--year",
        choices=("current", "previous"),
        help="Preferred column in the PDF tables (overrides config.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite non-empty cells in columns G/H/I.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging for troubleshooting.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    try:
        config = load_config(DEFAULT_CONFIG_PATH)
    except Exception as exc:  # pragma: no cover - fatal configuration issues
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
        logging.error("Failed to load configuration: %s", exc)
        return 2

    log_path = Path(config.get("log_path", DEFAULT_LOG_PATH))
    logger = configure_logging(debug=args.debug, log_path=log_path)
    logger.debug("Configuration loaded from %s", DEFAULT_CONFIG_PATH)

    try:
        return run(args=args, config=config, logger=logger)
    except DependencyError as exc:
        logger.error("%s", exc)
        return 2
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except Exception:  # pragma: no cover - safeguard against unexpected failures
        logger.exception("Unhandled error during processing")
        return 1


def run(*, args: argparse.Namespace, config: Dict[str, object], logger: logging.Logger) -> int:
    config = normalise_config(config)
    year_preference = args.year or config.get("year_preference", "current")
    overwrite_nonempty = args.force or config.get("overwrite_nonempty", False)
    config["year_preference"] = year_preference
    config["overwrite_nonempty"] = overwrite_nonempty

    excel_path = Path(args.excel).expanduser()
    if not excel_path.is_absolute():
        excel_path = Path.cwd() / excel_path
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    configure_dependencies(config, logger=logger)

    workbook, sheet, rows = excel_io.read_rows(excel_path)
    excel_io.ensure_result_columns(sheet)

    stats = ProcessingStats(total_rows=len(rows))
    report_entries: List[ReportEntry] = []

    engine = ocr_engine.OCREngine(config)
    pdf_idx = pdf_index.build_index(
        period_preference=year_preference,
        ocr_provider=engine,
        logger=logger,
    )

    for row in rows:
        if not row.mb:
            entry = ReportEntry(mb=row.mb or "<missing>")
            entry.notes.append("Excel row is missing Матични број")
            report_entries.append(entry)
            continue

        stats.rows_with_mb += 1
        entry = ReportEntry(mb=row.mb)
        entry.notes.extend(_collect_index_notes(pdf_idx, row.mb))

        existing = excel_io.get_existing_values(sheet, row.row_index)
        needs_field = {
            key: bool(overwrite_nonempty) or not _has_value(existing.get(key))
            for key in excel_io.RESULT_COLUMNS
        }
        skipped_fields = [key for key, needed in needs_field.items() if not needed]
        stats.values_skipped_existing += len(skipped_fields)

        required_fields = [key for key, needed in needs_field.items() if needed]
        if not required_fields and not entry.notes:
            continue

        row_updates: Dict[str, object] = {}

        if needs_field.get("revenue"):
            value, notes, missing = _extract_single_field(
                mb=row.mb,
                form_type="bu",
                field="revenue",
                index=pdf_idx,
                engine=engine,
                year_preference=year_preference,
                config=config,
                logger=logger,
            )
            entry.notes.extend(notes)
            if value is not None:
                row_updates["revenue"] = value
            if missing:
                entry.missing_fields.extend(missing)
                entry.missing_bu = True
                for field in missing:
                    stats.missing_fields[field] += 1

        bs_fields = [field for field in ("assets", "capital_loss") if needs_field.get(field)]
        if bs_fields:
            values, notes, missing = _extract_multiple_fields(
                mb=row.mb,
                form_type="bs",
                fields=bs_fields,
                index=pdf_idx,
                engine=engine,
                year_preference=year_preference,
                config=config,
                logger=logger,
            )
            entry.notes.extend(notes)
            row_updates.update(values)
            if missing:
                entry.missing_fields.extend(missing)
                entry.missing_bs = True
                for field in missing:
                    stats.missing_fields[field] += 1

        if row_updates:
            updates = excel_io.write_result_row(
                sheet,
                row_index=row.row_index,
                values=row_updates,
                overwrite_nonempty=overwrite_nonempty,
            )
            written = sum(1 for key in row_updates if updates.get(key))
            if written:
                stats.rows_with_updates += 1
                stats.values_written += written

        if entry.missing_bu or entry.missing_bs or entry.missing_fields or entry.notes:
            report_entries.append(entry)

    excel_io.save_workbook(workbook, excel_path)
    write_report(report_entries, REPORT_PATH)
    log_summary(logger, stats, report_entries)
    return 0


def normalise_config(config: Dict[str, object]) -> Dict[str, object]:
    result = dict(config)
    result.setdefault("year_preference", "current")
    result["year_preference"] = normalise_year(result.get("year_preference"))
    result["overwrite_nonempty"] = _to_bool(result.get("overwrite_nonempty", False))
    if "page_limit" in result:
        result["page_limit"] = _to_int(result.get("page_limit"))
    return result


def configure_logging(*, debug: bool, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(level)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.captureWarnings(True)
    return logger


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml must define a mapping")
    return data


def configure_dependencies(config: Dict[str, object], *, logger: logging.Logger) -> None:
    config["tesseract_path"] = resolve_tesseract_path(config.get("tesseract_path"))
    config["poppler_bin_dir"] = resolve_poppler_path(config.get("poppler_bin_dir"))
    logger.debug("Tesseract binary: %s", config["tesseract_path"])
    logger.debug("Poppler bin directory: %s", config["poppler_bin_dir"])


def resolve_tesseract_path(value: Optional[object]) -> str:
    candidates: List[Path] = []
    if value:
        candidates.append(Path(str(value)))
    which = shutil.which("tesseract")
    if which:
        candidates.append(Path(which))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise DependencyError("Tesseract executable not found. Update config.yaml or adjust PATH.")


def resolve_poppler_path(value: Optional[object]) -> str:
    candidates: List[Path] = []
    if value:
        candidates.append(Path(str(value)))
    which = shutil.which("pdftoppm")
    if which:
        candidates.append(Path(which).parent)
    for candidate in candidates:
        if candidate.exists():
            path = candidate if candidate.is_dir() else candidate.parent
            return str(path)
    raise DependencyError("Poppler binaries not found. Update config.yaml or adjust PATH.")


def _extract_single_field(
    *,
    mb: str,
    form_type: str,
    field: str,
    index: pdf_index.PdfIndex,
    engine: ocr_engine.OCREngine,
    year_preference: str,
    config: Dict[str, object],
    logger: logging.Logger,
) -> Tuple[Optional[object], List[str], List[str]]:
    entry = index.get_latest(mb, form_type, preferred_period=year_preference)
    notes: List[str] = []
    missing: List[str] = []

    if not entry:
        notes.append(f"No {form_type.upper()} PDF found for MB {mb}")
        missing.append(field)
        return None, notes, missing

    try:
        ocr_result = engine.process(entry.path)
    except Exception as exc:  # pragma: no cover - handled by OCR engine tests
        logger.exception("OCR failed for %s", entry.path)
        notes.append(f"OCR failed for {entry.path}: {exc}")
        missing.append(field)
        return None, notes, missing

    extractor = _resolve_extractor(form_type, field)
    if extractor is None:
        notes.append(f"Extractor for {field} is not implemented")
        missing.append(field)
        return None, notes, missing

    value, extractor_notes, meta = _invoke_extractor(
        extractor,
        ocr_result,
        year_preference,
        config,
        logger,
    )
    notes.extend(extractor_notes)
    if meta.get("has_errors") or not meta.get("has_value"):
        missing.append(field)
        return None, notes, missing
    return value, notes, missing


def _extract_multiple_fields(
    *,
    mb: str,
    form_type: str,
    fields: Iterable[str],
    index: pdf_index.PdfIndex,
    engine: ocr_engine.OCREngine,
    year_preference: str,
    config: Dict[str, object],
    logger: logging.Logger,
) -> Tuple[Dict[str, object], List[str], List[str]]:
    field_list = list(fields)
    entry = index.get_latest(mb, form_type, preferred_period=year_preference)
    notes: List[str] = []
    missing: List[str] = []
    values: Dict[str, object] = {}

    if not entry:
        notes.append(f"No {form_type.upper()} PDF found for MB {mb}")
        missing.extend(field_list)
        return values, notes, missing

    try:
        ocr_result = engine.process(entry.path)
    except Exception as exc:  # pragma: no cover - handled by OCR engine tests
        logger.exception("OCR failed for %s", entry.path)
        notes.append(f"OCR failed for {entry.path}: {exc}")
        missing.extend(field_list)
        return values, notes, missing

    for field in field_list:
        extractor = _resolve_extractor(form_type, field)
        if extractor is None:
            notes.append(f"Extractor for {field} is not implemented")
            missing.append(field)
            continue
        value, extractor_notes, meta = _invoke_extractor(
            extractor,
            ocr_result,
            year_preference,
            config,
            logger,
        )
        notes.extend(extractor_notes)
        if meta.get("has_errors") or not meta.get("has_value"):
            missing.append(field)
        else:
            values[field] = value

    return values, notes, missing


def _resolve_extractor(form_type: str, field: str):
    module = extract_bu if form_type == "bu" else extract_bs
    if module is None:
        return None
    for name in EXTRACTOR_NAMES.get(form_type, {}).get(field, ()):  # type: ignore[index]
        func = getattr(module, name, None)
        if callable(func):
            return func
    return None


def _invoke_extractor(func, ocr_result, year_preference: str, config: Dict[str, object], logger: logging.Logger):
    try:
        kwargs = {}
        signature = inspect.signature(func)
        if "year_preference" in signature.parameters:
            kwargs["year_preference"] = year_preference
        if "config" in signature.parameters:
            kwargs["config"] = config
        result = func(ocr_result, **kwargs)
    except Exception as exc:  # pragma: no cover - extractor failure path
        logger.exception("Extractor %s failed", getattr(func, "__name__", func))
        return None, [f"Extractor error: {exc}"], {
            "has_errors": True,
            "has_warnings": False,
            "has_value": False,
        }
    value, notes, meta = _coerce_extractor_result(result)
    return value, notes, meta


def _coerce_extractor_result(result) -> Tuple[Optional[object], List[str], Dict[str, bool]]:
    notes: List[str] = []
    has_errors = False
    has_warnings = False

    value: Any = result

    if isinstance(result, ExtractionResult):
        value = result.value
        if result.errors:
            has_errors = True
            notes.extend(_format_extraction_messages("ERROR", result.errors))
        if result.warnings:
            has_warnings = True
            notes.extend(_format_extraction_messages("WARNING", result.warnings))
    elif isinstance(result, dict):
        value = result.get("value")
        extra = result.get("notes") or result.get("note")
        notes.extend(_ensure_note_list(extra))
        if result.get("errors"):
            has_errors = True
            notes.extend(_format_message_entries("ERROR", result.get("errors")))
        if result.get("warnings"):
            has_warnings = True
            notes.extend(_format_message_entries("WARNING", result.get("warnings")))
    elif isinstance(result, tuple):
        value = result[0] if result else None
        if len(result) > 1:
            notes.extend(_ensure_note_list(result[1]))
        if len(result) > 2 and result[2]:
            has_errors = True
            notes.extend(_format_message_entries("ERROR", result[2]))
    else:
        value = getattr(result, "value", value)
        notes.extend(_ensure_note_list(getattr(result, "notes", None)))
        errors = getattr(result, "errors", None)
        warnings = getattr(result, "warnings", None)
        if errors:
            has_errors = True
            notes.extend(_format_message_entries("ERROR", errors))
        if warnings:
            has_warnings = True
            notes.extend(_format_message_entries("WARNING", warnings))

    has_value = _has_value(value)
    if not has_value:
        value = None

    meta = {
        "has_errors": has_errors,
        "has_warnings": has_warnings,
        "has_value": has_value,
    }
    return value, notes, meta


def _format_extraction_messages(level: str, messages: Iterable[ExtractionMessage]) -> List[str]:
    formatted: List[str] = []
    for message in messages:
        context = ""
        if message.context:
            details = ", ".join(f"{key}={value}" for key, value in sorted(message.context.items()))
            context = f" ({details})"
        formatted.append(f"{level} {message.code}: {message.message}{context}")
    return formatted


def _format_message_entries(level: str, messages: Any) -> List[str]:
    if messages is None:
        return []
    if isinstance(messages, (list, tuple, set)):
        items = list(messages)
    else:
        items = [messages]

    formatted: List[str] = []
    for item in items:
        if isinstance(item, ExtractionMessage):
            formatted.extend(_format_extraction_messages(level, [item]))
            continue
        if isinstance(item, dict):
            code = item.get("code", "unknown")
            message = item.get("message") or item.get("text") or str(item)
            context = item.get("context") or {}
            context_suffix = ""
            if context:
                details = ", ".join(f"{key}={value}" for key, value in sorted(context.items()))
                context_suffix = f" ({details})"
            formatted.append(f"{level} {code}: {message}{context_suffix}")
            continue
        formatted.append(f"{level}: {item}")
    return formatted


def _ensure_note_list(note) -> List[str]:
    if note is None:
        return []
    if isinstance(note, (list, tuple, set)):
        return [str(item) for item in note if item]
    return [str(note)]


def _collect_index_notes(index: pdf_index.PdfIndex, mb: str) -> List[str]:
    notes: List[str] = []
    notes_map = getattr(index, "report_notes", {})
    if mb in notes_map:
        for messages in notes_map[mb].values():
            notes.extend(messages)
    return notes


def write_report(entries: List[ReportEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["MB", "missing_bu", "missing_bs", "missing_fields", "notes"])
        for entry in entries:
            writer.writerow(entry.to_csv_row())


def log_summary(logger: logging.Logger, stats: ProcessingStats, report_entries: List[ReportEntry]) -> None:
    logger.info(
        "Processed %s rows (%s with Matični broj). Updated %s rows, wrote %s values (skipped %s existing).",
        stats.total_rows,
        stats.rows_with_mb,
        stats.rows_with_updates,
        stats.values_written,
        stats.values_skipped_existing,
    )
    if stats.missing_fields:
        summary = ", ".join(f"{field}:{count}" for field, count in stats.missing_fields.items())
        logger.info("Missing fields: %s", summary)
    if report_entries:
        logger.info("Report generated at %s", REPORT_PATH)


def _has_value(value: Optional[object]) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _to_bool(value: Optional[object]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _to_int(value: Optional[object]) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return int(str(value))


def normalise_year(value: Optional[object]) -> str:
    mapping = {
        "текућа": "current",
        "tekuca": "current",
        "current": "current",
        "претходна": "previous",
        "prethodna": "previous",
        "previous": "previous",
    }
    if value is None:
        return "current"
    return mapping.get(str(value).strip().lower(), "current")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
