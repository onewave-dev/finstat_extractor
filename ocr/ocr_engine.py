"""OCR engine implementation based on Poppler rendering and Tesseract."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pdf2image import convert_from_path, pdfinfo_from_path
import pytesseract
from pytesseract import Output, TesseractError


DEFAULT_CACHE_PATH = Path("cache") / "cache.sqlite"
DEFAULT_LOG_PATH = Path("logs") / "finstat_extractor.log"


class OCRProcessingError(RuntimeError):
    """Raised when OCR processing fails for a PDF document."""


@dataclass
class OcrPage:
    page_number: int
    text: str
    tsv: List[Dict[str, str]]


@dataclass
class OcrResult:
    pdf_path: str
    pdf_hash: str
    text: str
    pages: List[OcrPage]

    def to_dict(self) -> Dict[str, object]:
        return {
            "pdf_path": self.pdf_path,
            "pdf_hash": self.pdf_hash,
            "text": self.text,
            "pages": [asdict(page) for page in self.pages],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "OcrResult":
        pages = [OcrPage(**page_dict) for page_dict in data.get("pages", [])]
        return cls(
            pdf_path=str(data.get("pdf_path", "")),
            pdf_hash=str(data["pdf_hash"]),
            text=str(data.get("text", "")),
            pages=pages,
        )


class OCREngine:
    """High level OCR engine that caches results in SQLite."""

    def __init__(
        self,
        config: Dict[str, object],
        cache_path: Path = DEFAULT_CACHE_PATH,
        log_path: Path = DEFAULT_LOG_PATH,
    ) -> None:
        self.config = config
        self.cache_path = Path(cache_path)
        self.log_path = Path(log_path)

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = self._configure_logger()

        self._ensure_cache_schema()
        self._configure_tesseract()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, pdf_path: Path) -> OcrResult:
        """Perform OCR on the provided PDF file with caching."""

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        pdf_hash = self._compute_pdf_hash(pdf_path)
        cached = self._load_from_cache(pdf_hash)
        if cached is not None:
            self.logger.info("Using cached OCR for %s", pdf_path)
            return cached

        try:
            images = self._render_pdf(pdf_path)
            result = self._run_tesseract(pdf_path, pdf_hash, images)
        except Exception as exc:
            self.logger.exception("Failed OCR for %s", pdf_path)
            raise OCRProcessingError(str(exc)) from exc

        self._store_in_cache(result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _configure_logger(self) -> logging.Logger:
        logger = logging.getLogger("finstat_extractor")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.log_path, encoding="utf-8")
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.propagate = False
        return logger

    def _configure_tesseract(self) -> None:
        tesseract_path = self.config.get("tesseract_path")
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)

    def _compute_pdf_hash(self, pdf_path: Path) -> str:
        hasher = hashlib.sha1()
        stat = pdf_path.stat()
        hasher.update(str(stat.st_size).encode())
        hasher.update(str(int(stat.st_mtime)).encode())
        with pdf_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1_048_576), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _render_pdf(self, pdf_path: Path) -> List["Image.Image"]:
        dpi = int(self.config.get("dpi", 300))
        page_limit = int(self.config.get("page_limit", 2))
        poppler_dir = self.config.get("poppler_bin_dir")

        info = pdfinfo_from_path(str(pdf_path), poppler_path=poppler_dir)
        page_count = int(info.get("Pages", 0))
        if page_count == 0:
            self.logger.warning("PDF has zero pages: %s", pdf_path)
            return []

        last_page = min(page_limit, page_count) if page_limit else page_count

        kwargs = {
            "dpi": dpi,
            "first_page": 1,
            "poppler_path": poppler_dir,
        }
        if last_page:
            kwargs["last_page"] = last_page

        images = convert_from_path(str(pdf_path), **kwargs)
        return images

    def _run_tesseract(
        self,
        pdf_path: Path,
        pdf_hash: str,
        images: Iterable["Image.Image"],
    ) -> OcrResult:
        langs = self.config.get("ocr_langs", "srp+srp_latn")

        page_results: List[OcrPage] = []
        full_text_parts: List[str] = []

        for page_number, image in enumerate(images, start=1):
            try:
                text = pytesseract.image_to_string(image, lang=langs)
                tsv_string = pytesseract.image_to_data(
                    image, lang=langs, output_type=Output.STRING
                )
            except TesseractError as exc:
                self.logger.exception(
                    "Tesseract failed on %s page %s", pdf_path, page_number
                )
                raise OCRProcessingError(str(exc)) from exc

            page_results.append(
                OcrPage(
                    page_number=page_number,
                    text=text,
                    tsv=list(self._parse_tsv(tsv_string)),
                )
            )
            full_text_parts.append(text)

        result = OcrResult(
            pdf_path=str(pdf_path),
            pdf_hash=pdf_hash,
            text="\n".join(full_text_parts),
            pages=page_results,
        )
        return result

    def _parse_tsv(self, tsv_string: str) -> Iterable[Dict[str, str]]:
        lines = [line for line in tsv_string.splitlines() if line.strip()]
        if not lines:
            return []

        header = lines[0].split("\t")
        for row in lines[1:]:
            values = row.split("\t")
            if len(values) != len(header):
                continue
            yield {header[i]: values[i] for i in range(len(header))}

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _ensure_cache_schema(self) -> None:
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ocr_cache (
                    pdf_hash TEXT PRIMARY KEY,
                    pdf_path TEXT,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _load_from_cache(self, pdf_hash: str) -> Optional[OcrResult]:
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute(
                "SELECT payload FROM ocr_cache WHERE pdf_hash = ?", (pdf_hash,)
            )
            row = cursor.fetchone()
        if not row:
            return None
        payload = json.loads(row[0])
        return OcrResult.from_dict(payload)

    def _store_in_cache(self, result: OcrResult) -> None:
        payload = json.dumps(result.to_dict())
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                "REPLACE INTO ocr_cache(pdf_hash, pdf_path, payload, created_at)"
                " VALUES (?, ?, ?, ?)",
                (
                    result.pdf_hash,
                    result.pdf_path,
                    payload,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()


__all__ = ["OCREngine", "OcrResult", "OcrPage", "OCRProcessingError"]

