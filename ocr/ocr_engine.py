"""OCR engine implementation based on Poppler rendering and Tesseract."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from pdf2image import convert_from_path, pdfinfo_from_path
import pytesseract
from pytesseract import Output, TesseractError


DEFAULT_CACHE_PATH = Path("cache") / "cache.sqlite"
DEFAULT_LOG_PATH = Path("logs") / "finstat_extractor.log"
DEFAULT_PAGE_LIMIT = 30


class OCRProcessingError(RuntimeError):
    """Raised when OCR processing fails for a PDF document."""


@dataclass
class OcrPage:
    page_number: int
    text: str
    tsv: List[Dict[str, str]]
    confidence: Optional[float] = None
    dpi: Optional[int] = None
    preprocessing_pipeline: Optional[List[str]] = None


@dataclass
class OcrResult:
    pdf_path: str
    pdf_hash: str
    text: str
    pages: List[OcrPage]
    average_confidence: Optional[float] = None
    text_source: str = "tesseract"

    def to_dict(self) -> Dict[str, object]:
        return {
            "pdf_path": self.pdf_path,
            "pdf_hash": self.pdf_hash,
            "text": self.text,
            "pages": [asdict(page) for page in self.pages],
            "average_confidence": self.average_confidence,
            "text_source": self.text_source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "OcrResult":
        pages: List[OcrPage] = []
        for page_dict in data.get("pages", []):
            if "preprocessing_pipeline" not in page_dict:
                page_dict = {**page_dict, "preprocessing_pipeline": None}
            if "confidence" not in page_dict:
                page_dict = {**page_dict, "confidence": None}
            if "dpi" not in page_dict:
                page_dict = {**page_dict, "dpi": None}
            pages.append(OcrPage(**page_dict))
        return cls(
            pdf_path=str(data.get("pdf_path", "")),
            pdf_hash=str(data["pdf_hash"]),
            text=str(data.get("text", "")),
            pages=pages,
            average_confidence=data.get("average_confidence"),
            text_source=str(data.get("text_source", "tesseract")),
        )


class OCREngine:
    """High level OCR engine that caches results in SQLite."""

    def __init__(
        self,
        config: Dict[str, object],
        cache_path: Path = DEFAULT_CACHE_PATH,
        log_path: Path = DEFAULT_LOG_PATH,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.cache_path = Path(cache_path)
        self.log_path = Path(log_path)

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logger or self._configure_logger()

        self._ensure_cache_schema()
        self._configure_tesseract()
        self._cv2_module = None

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

        plumber_pages = self._extract_pdf_text(pdf_path)
        page_limit = self._resolve_page_limit()
        if plumber_pages and len(plumber_pages) > page_limit:
            plumber_pages = plumber_pages[:page_limit]

        try:
            primary_dpi = self._resolve_dpi(None)
            images = self._render_pdf(pdf_path, dpi_override=primary_dpi)
            if not images and plumber_pages:
                text = "\n".join(plumber_pages)
                result = OcrResult(
                    pdf_path=str(pdf_path),
                    pdf_hash=pdf_hash,
                    text=text,
                    pages=[],
                    average_confidence=None,
                    text_source="pdfplumber",
                )
            else:
                result = self._run_tesseract(
                    pdf_path,
                    pdf_hash,
                    images,
                    primary_dpi,
                    plumber_pages=plumber_pages,
                )

            confidence_threshold = self._get_confidence_threshold()
            if (
                images
                and confidence_threshold is not None
                and (result.average_confidence or 0.0) < confidence_threshold
            ):
                for retry_dpi in self._iter_retry_dpis(primary_dpi):
                    self.logger.info(
                        "Retrying OCR for %s at %s dpi due to low confidence %.2f < %.2f",
                        pdf_path,
                        retry_dpi,
                        result.average_confidence or 0.0,
                        confidence_threshold,
                    )
                    retry_images = self._render_pdf(pdf_path, dpi_override=retry_dpi)
                    retry_result = self._run_tesseract(
                        pdf_path,
                        pdf_hash,
                        retry_images,
                        retry_dpi,
                        plumber_pages=plumber_pages,
                    )
                    new_conf = retry_result.average_confidence or 0.0
                    if (
                        result.average_confidence is None
                        or new_conf > (result.average_confidence or 0.0)
                    ):
                        result = retry_result
                    if (
                        confidence_threshold is not None
                        and new_conf >= confidence_threshold
                    ):
                        break
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

    def _resolve_dpi(self, dpi_override: Optional[int]) -> int:
        if dpi_override is not None:
            try:
                dpi = int(dpi_override)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid dpi override %r. Falling back to configured dpi.",
                    dpi_override,
                )
            else:
                if dpi > 0:
                    return dpi
                self.logger.warning(
                    "DPI override must be positive. Falling back to configured dpi.",
                )
        configured_dpi = self.config.get("dpi", 300)
        try:
            dpi = int(configured_dpi)
        except (TypeError, ValueError):
            self.logger.warning(
                "Invalid dpi=%r in config; defaulting to 300 dpi.", configured_dpi
            )
            return 300
        if dpi <= 0:
            self.logger.warning(
                "Configured dpi=%s is not positive; defaulting to 300 dpi.", dpi
            )
            return 300
        return dpi

    def _resolve_page_limit(self) -> int:
        page_limit_raw = self.config.get("page_limit")
        page_limit = DEFAULT_PAGE_LIMIT
        if page_limit_raw is not None:
            try:
                parsed_limit = int(page_limit_raw)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid page_limit=%r in config; defaulting to %s pages",
                    page_limit_raw,
                    DEFAULT_PAGE_LIMIT,
                )
            else:
                if parsed_limit > 0:
                    page_limit = parsed_limit
                else:
                    self.logger.warning(
                        "Configured page_limit=%s is not positive; defaulting to %s pages",
                        parsed_limit,
                        DEFAULT_PAGE_LIMIT,
                    )
        return page_limit

    def _get_confidence_threshold(self) -> Optional[float]:
        threshold = self.config.get("confidence_threshold")
        if threshold is None:
            return None
        try:
            value = float(threshold)
        except (TypeError, ValueError):
            self.logger.warning(
                "Invalid confidence_threshold=%r in config; ignoring.", threshold
            )
            return None
        if value <= 0:
            self.logger.warning(
                "Configured confidence_threshold=%s is not positive; ignoring.",
                value,
            )
            return None
        return value

    def _iter_retry_dpis(self, primary_dpi: int) -> Iterable[int]:
        dpi_retry = self.config.get("dpi_retry")
        if dpi_retry is None:
            return []

        if isinstance(dpi_retry, (list, tuple)):
            values: Sequence[object] = dpi_retry
        else:
            values = [dpi_retry]

        seen = set()
        for candidate in values:
            try:
                dpi = int(candidate)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid dpi_retry value %r; expected an integer.", candidate
                )
                continue
            if dpi <= 0 or dpi == primary_dpi or dpi in seen:
                continue
            seen.add(dpi)
            yield dpi

    def _get_psm_value(self, key: str, default: int) -> int:
        value = self.config.get(key)
        if value is None:
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            self.logger.warning("Invalid %s=%r; falling back to %s", key, value, default)
            return default
        if parsed <= 0:
            self.logger.warning("Configured %s=%s is not positive; using %s", key, parsed, default)
            return default
        return parsed

    def _cv2(self):
        if self._cv2_module is None:
            try:
                import cv2  # pylint: disable=import-outside-toplevel
            except ImportError as exc:  # pragma: no cover - exercised via error handling tests
                message = (
                    "OpenCV (cv2) is required for image preprocessing. Install it via "
                    "`pip install opencv-python`."
                )
                self.logger.error(message)
                raise OCRProcessingError(message) from exc

            self._cv2_module = cv2
        return self._cv2_module

    def _compute_pdf_hash(self, pdf_path: Path) -> str:
        hasher = hashlib.sha1()
        stat = pdf_path.stat()
        hasher.update(str(stat.st_size).encode())
        hasher.update(str(int(stat.st_mtime)).encode())
        with pdf_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1_048_576), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _render_pdf(self, pdf_path: Path, dpi_override: Optional[int] = None) -> List[Image.Image]:
        dpi = self._resolve_dpi(dpi_override)
        page_limit = self._resolve_page_limit()
        poppler_dir = self.config.get("poppler_bin_dir")

        info = pdfinfo_from_path(str(pdf_path), poppler_path=poppler_dir)
        page_count = int(info.get("Pages", 0))
        if page_count == 0:
            self.logger.warning("PDF has zero pages: %s", pdf_path)
            return []

        last_page = min(page_limit, page_count)

        kwargs = {
            "dpi": dpi,
            "first_page": 1,
            "poppler_path": poppler_dir,
            "last_page": last_page,
        }

        images = convert_from_path(str(pdf_path), **kwargs)
        if page_count > page_limit:
            self.logger.warning(
                "Processed only the first %s pages out of %s in %s due to the configured page limit.",
                page_limit,
                page_count,
                pdf_path,
            )
        return images

    def _run_tesseract(
        self,
        pdf_path: Path,
        pdf_hash: str,
        images: Iterable[Image.Image],
        dpi: int,
        plumber_pages: Optional[List[str]] = None,
    ) -> OcrResult:
        langs = self.config.get("ocr_langs", "srp+srp_latn")
        tsv_psm = self._get_psm_value("tsv_psm", 4)
        text_psm = self._get_psm_value("text_psm", 6)

        plumber_pages = plumber_pages or []

        page_results: List[OcrPage] = []
        full_text_parts: List[str] = []
        plumber_used = 0

        diagnostics_dir: Optional[Path] = None
        save_pages = self._should_save_preprocessed_pages()
        if save_pages:
            diagnostics_dir = self._get_diagnostics_dir()

        for page_number, image in enumerate(images, start=1):
            processed_array, pipeline_steps = self._apply_preprocessing_pipeline(image)
            processed_image = self._array_to_pil(processed_array)

            if save_pages and diagnostics_dir is not None and pipeline_steps:
                self._save_diagnostic_image(
                    processed_image,
                    diagnostics_dir,
                    pdf_path,
                    page_number,
                    dpi,
                    "preprocessed",
                )

            try:
                text = pytesseract.image_to_string(
                    processed_image,
                    lang=langs,
                    config=f"--psm {text_psm}",
                )
                tsv_string = pytesseract.image_to_data(
                    processed_image,
                    lang=langs,
                    config=f"--psm {tsv_psm}",
                    output_type=Output.STRING,
                )
            except TesseractError as exc:
                self.logger.exception(
                    "Tesseract failed on %s page %s", pdf_path, page_number
                )
                raise OCRProcessingError(str(exc)) from exc

            tsv_rows = list(self._parse_tsv(tsv_string))
            page_confidence = self._compute_page_confidence(tsv_rows)
            if page_confidence is not None:
                self.logger.info(
                    "Page %s OCR confidence at %s dpi: %.2f",
                    page_number,
                    dpi,
                    page_confidence,
                )

            plumber_text = ""
            if page_number <= len(plumber_pages):
                plumber_text = plumber_pages[page_number - 1] or ""

            if plumber_text.strip():
                text_output = plumber_text
                plumber_used += 1
            else:
                text_output = text

            page_results.append(
                OcrPage(
                    page_number=page_number,
                    text=text_output,
                    tsv=tsv_rows,
                    confidence=page_confidence,
                    dpi=dpi,
                    preprocessing_pipeline=pipeline_steps if pipeline_steps else None,
                )
            )
            full_text_parts.append(text_output)

        overall_confidence = self._calculate_overall_confidence(page_results)
        text_source = "tesseract"
        if plumber_used and plumber_used == len(page_results):
            text_source = "pdfplumber"
        elif plumber_used:
            text_source = "pdfplumber+tesseract"

        result = OcrResult(
            pdf_path=str(pdf_path),
            pdf_hash=pdf_hash,
            text="\n".join(full_text_parts),
            pages=page_results,
            average_confidence=overall_confidence,
            text_source=text_source,
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

    def _apply_preprocessing_pipeline(
        self,
        image: Image.Image,
        pipeline_config: Optional[Dict[str, object]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        config = pipeline_config
        if config is None or not isinstance(config, dict):
            config = self.config.get("preprocessing", {}) or {}

        if not config.get("enabled", True):
            return np.array(image), []

        pipeline = config.get("pipeline")
        if not isinstance(pipeline, Sequence):
            pipeline = [
                "grayscale",
                "denoise",
                "adaptive_threshold",
                "morphology",
                "deskew",
            ]

        cv2 = self._cv2()
        working = np.array(image)
        executed: List[str] = []

        for step in pipeline:
            step_name = str(step).lower()
            if step_name == "grayscale":
                if working.ndim == 3:
                    working = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)
                executed.append("grayscale")
            elif step_name == "denoise":
                denoise_cfg = config.get("denoise", {}) or {}
                method = str(
                    denoise_cfg.get(
                        "method", config.get("denoise_method", "median")
                    )
                ).lower()
                if method == "median":
                    kernel = int(denoise_cfg.get("median_kernel", 3))
                    if kernel % 2 == 0:
                        kernel += 1
                    kernel = max(3, kernel)
                    working = cv2.medianBlur(working, kernel)
                    executed.append("denoise:median")
                elif method == "bilateral":
                    d = int(denoise_cfg.get("bilateral_d", denoise_cfg.get("d", 9)))
                    sigma_color = float(
                        denoise_cfg.get(
                            "sigma_color",
                            denoise_cfg.get("bilateral_sigma_color", 75),
                        )
                    )
                    sigma_space = float(
                        denoise_cfg.get(
                            "sigma_space",
                            denoise_cfg.get("bilateral_sigma_space", 75),
                        )
                    )
                    working = cv2.bilateralFilter(working, d, sigma_color, sigma_space)
                    executed.append("denoise:bilateral")
                else:
                    continue
            elif step_name == "adaptive_threshold":
                thresh_cfg = config.get("adaptive_threshold", {}) or {}
                if working.ndim == 3:
                    working = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)
                block_size = int(thresh_cfg.get("block_size", 31))
                if block_size % 2 == 0:
                    block_size += 1
                block_size = max(3, block_size)
                c_value = float(thresh_cfg.get("c", thresh_cfg.get("C", 5)))
                method = str(thresh_cfg.get("method", "gaussian")).lower()
                adaptive_method = (
                    cv2.ADAPTIVE_THRESH_MEAN_C
                    if method == "mean"
                    else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                )
                working = cv2.adaptiveThreshold(
                    working,
                    255,
                    adaptive_method,
                    cv2.THRESH_BINARY,
                    block_size,
                    c_value,
                )
                executed.append(f"adaptive_threshold:{method}")
            elif step_name == "morphology":
                morph_cfg = config.get("morphology", {}) or {}
                operation = str(morph_cfg.get("operation", "open")).lower()
                kernel_size = int(morph_cfg.get("kernel_size", 3))
                kernel_size = max(1, kernel_size)
                iterations = int(morph_cfg.get("iterations", 1))
                iterations = max(1, iterations)
                if working.ndim == 3:
                    working = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (kernel_size, kernel_size)
                )
                op_map = {
                    "open": cv2.MORPH_OPEN,
                    "close": cv2.MORPH_CLOSE,
                    "erode": cv2.MORPH_ERODE,
                    "dilate": cv2.MORPH_DILATE,
                }
                op = op_map.get(operation)
                if op is None:
                    self.logger.warning(
                        "Unknown morphology operation %s; skipping", operation
                    )
                    continue
                working = cv2.morphologyEx(working, op, kernel, iterations=iterations)
                executed.append(f"morphology:{operation}")
            elif step_name == "deskew":
                working = self._deskew_image(working)
                executed.append("deskew")
            else:
                self.logger.debug("Unknown preprocessing step '%s'; skipping", step_name)

        return working, executed

    def _deskew_image(self, image_array: np.ndarray) -> np.ndarray:
        cv2 = self._cv2()
        if image_array.ndim == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        binary = cv2.bitwise_not(binary)
        coords = np.column_stack(np.where(binary > 0))
        if coords.size == 0:
            return image_array

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = gray.shape[:2]
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image_array,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return np.clip(rotated, 0, 255).astype("uint8")

    def _array_to_pil(self, array: np.ndarray) -> Image.Image:
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype("uint8")
        if array.ndim == 2:
            return Image.fromarray(array)
        return Image.fromarray(array)

    def _compute_page_confidence(
        self, tsv_rows: Sequence[Dict[str, str]]
    ) -> Optional[float]:
        confidences: List[float] = []
        for row in tsv_rows:
            conf_value = row.get("conf")
            if conf_value in (None, "", "-1"):
                continue
            try:
                conf = float(conf_value)
            except (TypeError, ValueError):
                continue
            confidences.append(conf)
        if not confidences:
            return None
        return float(sum(confidences) / len(confidences))

    def _calculate_overall_confidence(
        self, pages: Sequence[OcrPage]
    ) -> Optional[float]:
        confidences = [page.confidence for page in pages if page.confidence is not None]
        if not confidences:
            return None
        return float(sum(confidences) / len(confidences))

    def _diagnostics_config(self) -> Dict[str, object]:
        diagnostics = self.config.get("diagnostics")
        if isinstance(diagnostics, dict):
            return diagnostics
        return {}

    def _diagnostics_enabled(self) -> bool:
        diagnostics = self._diagnostics_config()
        return bool(diagnostics.get("enabled"))

    def _should_save_preprocessed_pages(self) -> bool:
        diagnostics = self._diagnostics_config()
        return bool(
            self._diagnostics_enabled() and diagnostics.get("save_preprocessed_pages")
        )

    def _should_save_numeric_crops(self) -> bool:
        diagnostics = self._diagnostics_config()
        return bool(
            self._diagnostics_enabled() and diagnostics.get("save_numeric_crops")
        )

    def _get_diagnostics_dir(self) -> Path:
        diagnostics = self._diagnostics_config()
        directory = diagnostics.get("dir")
        if directory:
            path = Path(directory)
        else:
            path = self.log_path.parent / "diagnostics"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_diagnostic_image(
        self,
        image: Image.Image,
        directory: Path,
        pdf_path: Path,
        page_number: int,
        dpi: int,
        suffix: str,
    ) -> None:
        filename = (
            f"{pdf_path.stem.replace(' ', '_')}_page{page_number}_dpi{dpi}_{suffix}.png"
        )
        target = directory / filename
        try:
            image.save(target)
        except Exception as exc:  # pragma: no cover - diagnostic helper
            self.logger.warning(
                "Failed to save diagnostic image %s: %s", target, exc
            )

    def _save_numeric_crop(
        self,
        image: Image.Image,
        directory: Path,
        label: str,
    ) -> None:
        filename = f"{label}_{uuid.uuid4().hex}.png"
        target = directory / filename
        try:
            image.save(target)
        except Exception as exc:  # pragma: no cover - diagnostic helper
            self.logger.warning(
                "Failed to save numeric diagnostic crop %s: %s", target, exc
            )

    def _extract_pdf_text(self, pdf_path: Path) -> List[str]:
        if not self.config.get("enable_pdf_text", True):
            return []

        texts: List[str] = []
        try:
            import pdfplumber  # pylint: disable=import-outside-toplevel

            with pdfplumber.open(str(pdf_path)) as pdf:
                page_limit = self._resolve_page_limit()
                pages = pdf.pages[:page_limit]
                for index, page in enumerate(pages, start=1):
                    try:
                        page_text = page.extract_text() or ""
                    except Exception as exc:  # pragma: no cover - third-party behavior
                        self.logger.debug(
                            "pdfplumber failed on %s page %s: %s",
                            pdf_path,
                            index,
                            exc,
                        )
                        page_text = ""
                    texts.append(page_text.strip())
        except Exception as exc:
            self.logger.debug("pdfplumber could not open %s: %s", pdf_path, exc)
            return []

        if any(texts):
            self.logger.info(
                "Extracted textual content via pdfplumber for %s", pdf_path
            )
        return texts

    def _normalize_bbox(
        self, bbox: Sequence[int], image_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        if len(bbox) != 4:
            raise ValueError("bbox must contain four values")

        width, height = image_size
        x0 = int(round(bbox[0]))
        y0 = int(round(bbox[1]))
        third = int(round(bbox[2]))
        fourth = int(round(bbox[3]))

        if third > x0 and fourth > y0:
            # Treat as x2, y2
            x1 = min(third, width)
            y1 = min(fourth, height)
            w = x1 - x0
            h = y1 - y0
        else:
            w = third
            h = fourth
        if w <= 0:
            w = 1
        if h <= 0:
            h = 1
        x0 = max(0, min(x0, width - 1))
        y0 = max(0, min(y0, height - 1))
        w = min(w, width - x0)
        h = min(h, height - y0)
        return x0, y0, w, h

    def ocr_numeric_region(
        self,
        image: Image.Image,
        bbox: Sequence[int],
        config: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))

        override_config = config or {}
        preprocessing_override = override_config.get("preprocessing")

        numeric_psm = self._get_psm_value("numeric_psm", 13)
        if "psm" in override_config:
            try:
                numeric_psm = int(override_config["psm"])
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid numeric region psm=%r; using default %s",
                    override_config["psm"],
                    numeric_psm,
                )

        langs = override_config.get(
            "langs",
            self.config.get("numeric_langs", self.config.get("ocr_langs", "eng")),
        )

        whitelist = override_config.get(
            "whitelist",
            "0123456789()-. ,",
        )

        x0, y0, w, h = self._normalize_bbox(bbox, image.size)
        crop_box = (x0, y0, x0 + w, y0 + h)
        crop = image.crop(crop_box)

        processed_array, pipeline_steps = self._apply_preprocessing_pipeline(
            crop, pipeline_config=preprocessing_override
        )
        processed_image = self._array_to_pil(processed_array)

        diagnostics_dir: Optional[Path] = None
        if self._should_save_numeric_crops():
            diagnostics_dir = self._get_diagnostics_dir()
            label = str(override_config.get("diagnostics_label", "numeric_crop"))
            self._save_numeric_crop(processed_image, diagnostics_dir, label)

        tess_config = (
            f"--oem 1 --psm {numeric_psm} -c tessedit_char_whitelist={whitelist}"
        )
        try:
            text = pytesseract.image_to_string(
                processed_image,
                lang=langs,
                config=tess_config,
            )
            tsv_string = pytesseract.image_to_data(
                processed_image,
                lang=langs,
                config=tess_config,
                output_type=Output.STRING,
            )
        except TesseractError as exc:
            self.logger.exception("Numeric OCR failed for bbox %s", bbox)
            raise OCRProcessingError(str(exc)) from exc

        tsv_rows = list(self._parse_tsv(tsv_string))
        confidence = self._compute_page_confidence(tsv_rows)
        if confidence is not None:
            self.logger.info(
                "Numeric OCR confidence %.2f for bbox %s", confidence, bbox
            )

        return {
            "text": text.strip(),
            "confidence": confidence,
            "tsv": tsv_rows,
            "bbox": (x0, y0, w, h),
            "preprocessing_pipeline": pipeline_steps if pipeline_steps else None,
            "psm": numeric_psm,
            "langs": langs,
        }

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

