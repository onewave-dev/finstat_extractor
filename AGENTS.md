# AGENTS.md — FinStat Extractor (ChatGPT Codex Agent)

> **Purpose:** Implement a local Windows-friendly tool that reads **Matični broj** values from an Excel file and fills columns **G/H/I** by extracting figures from Serbian financial statement PDFs (**Bilans uspeha** and **Bilans stanja**) using OCR.  
> **Key constraint:** PDFs are located **in the same folder** as the final `app.exe` (i.e., the working directory). Filenames are arbitrary; matching is done by **Matični broj inside the PDF**.  
> **Audience:** ChatGPT Codex (and human developers) to build a robust, portable utility.

---

## 1) Functional Scope

**Input**
- One Excel file (e.g., `UPITNIK_primer 1.xls`) with a column named **"Матични број"** (exact or near-match by case/spacing).
- A folder containing **PDF scans** of:
  - **Bilans uspeha** (BU) — must include a row labeled **"Пословни приходи"**.
  - **Bilans stanja** (BS) — must include rows **"Укупна актива"** and **"Губитак изнад висине капитала"**.
- All PDFs lie **beside** the executable (`app.exe`) in the **same directory**. The program assumes **current working directory** is where PDFs are.
- OCR prerequisites:
  - **Tesseract language packs**: `srp`, `srp_latn`, `eng` (installed locally and referenced via `config.yaml`).
  - **Python dependencies**: OpenCV (`opencv-python`), `numpy`, `Pillow`, `pdfplumber` (fallback for text PDFs), in addition to the existing stack (pytesseract, pdf2image, etc.).
  - **Processing pipeline**: two-pass OCR (page-level anchor detection + cropped numeric pass), OpenCV preprocessing (grayscale → blur → adaptive threshold → deskew), and multi-DPI retries when confidence falls below thresholds.

**Output**
- The same Excel file with columns **G/H/I** populated:
  - **G:** `Пословни приходи (000 РСД)` ← from **BU**.
  - **H:** `Укупна актива (000 РСД)` ← from **BS**.
  - **I:** `Губитак изнад висине капитала (000 РСД)` ← computed as `max(0, −AOP 0401)` and sourced from **BS** numeric cells.
- Diagnostic artifacts:
  - `report_missing.csv` — per Matični broj, which documents/fields were missing or unreadable.
  - `logs/finstat_extractor.log` — run-time log with INFO/WARN/ERROR.
- Optional cache DB `cache/cache.sqlite` to speed up re-runs.
- Persisted extraction diagnostics per value (`source_file`, `page_idx`, `method`, `confidence_avg`, numeric result) to aid QA and retries.

**Rules**
- Match PDFs to Excel **by Matični broj found *inside* each PDF** using OCR.
- Filenames are **not** used for matching.
- Values remain in **thousands of RSD** (as printed on forms).
- Prefer the **"Текућа година"** (current year) column; allow a switch to **"Претходна година"** via CLI/config.

---

## 2) Architecture (Implementation Plan)

### 2.1 Project Layout
```
finstat_extractor/
  app.py                  # CLI entrypoint; orchestration
  config.yaml             # OCR langs, poppler path, anchors/synonyms, year preference
  /io/
    excel_io.py           # read/write Excel; ensure columns G/H/I
  /index/
    pdf_index.py          # walk cwd, OCR, detect MB + form type -> index
  /ocr/
    ocr_engine.py         # pdf->images (poppler) -> tesseract(tsv+text) + caching
    anchors.py            # regex & label synonyms, header detection ("Текућа/Претходна година")
  /extract/
    bu.py                 # "Пословни приходи" value extraction (BU)
    bs.py                 # "Укупна актива", "Губитак изнад..." extraction (BS)
  /cache/
    cache.sqlite          # (created at runtime) OCR/parse cache by file hash
  /logs/
    finstat_extractor.log # (created at runtime)
  build.bat               # PyInstaller packaging script
  README.md               # End-user installation & usage
```

### 2.2 Data Flow
1. **Load Excel:** read column **"Матични број"**, normalize MB to 8–9 digits.
2. **Index PDFs (cwd):** for each `*.pdf`:
   - Render first 1–2 pages with **Poppler** (300 dpi baseline; allow multi-DPI retries on low confidence).
   - OCR with **Tesseract** (`srp+srp_latn+eng`) using a **two-pass pipeline**:
     - **Pass 1:** page-level anchor detection (full-page OCR → TSV with bbox coordinates).
     - **Pass 2:** cropped numeric regions reprocessed (potentially at higher DPI) for improved confidence.
   - Apply OpenCV preprocessing before each OCR call: grayscale, denoise/blur, adaptive thresholding, and deskew.
   - When PDFs are already text-based or OCR confidence remains low after retries, invoke **pdfplumber** as a fallback text extractor.
   - Detect **Matični broj** via regex (both Cyrillic and Latin spellings).
   - Classify document type:
     - BU if header/body matches `Билан(с|c)\s+успеха|Bilans\s+uspeha`.
     - BS if header/body matches `Билан(с|c)\s+стања|Bilans\s+stanja`.
   - Insert into index: `{ MB: { bu: [pdf_id...], bs: [pdf_id...] } }`. If multiple, pick the newest by `mtime` (tie-break by detected period if available).
3. **Extract Values:**
   - BU → find row anchor **"Пословни приходи"**; move horizontally to the **"Текућа година"** numeric cell (or "Претходна" if configured).
   - BS → find **"Укупна актива"** and **"Губитак изнад висине капитала"** similarly; compute **"Губитак изнад висине капитала"** as `max(0, −AOP 0401)` even if table shows alternative formatting.
   - Use TSV coordinates to search **right of the anchor within the same text-line band**; filter to the nearest numeric token inside the target column window.
   - Normalize numbers: remove thousand separators (`.`/space), convert commas to dot, keep sign, keep as thousands RSD.
   - Persist diagnostics (`source_file`, `page_idx`, OCR `method` used, `confidence_avg`, raw text) for each extracted numeric value.
4. **Write Excel:** ensure columns **G/H/I** exist with exact headers; write results per row/MB; leave blank if not found; don’t overwrite non-empty cells unless `--force` is passed; include diagnostic metadata alongside numeric values (e.g., via hidden columns or external report) for traceability.
5. **Reporting:** emit `report_missing.csv` for MBs with absent docs/anchors/numbers; log detailed context.

### 2.3 OCR Anchors & Regex
- Matični broj (case/space tolerant, Cyrillic/Latin):
  - `(?i)(ма[тт]ични|maticni|matični)\s*број|broj`
  - Capture: `([0-9]{8,9})`
- Form headers:
  - BU: `(?i)билан[сc]\s+успеха|bilans\s+uspeha`
  - BS: `(?i)билан[сc]\s+стања|bilans\s+stanja`
- Row anchors:
  - BU: `(?i)пословни\s+приходи`
  - BS-1: `(?i)укупна\s+актива`
  - BS-2: `(?i)губитак\s+изнад\s+висине\s+капитала`
- The extractor recognises uppercase, hyphenated and Latin variants of the row anchors (e.g. `POSLOVNI-PRIHODI`, `UKUPNA AKTIVA`, `КАПИТАЛ`) and has dedicated fallbacks for the AOP codes (`1001`, `0059`, `0401`). When a textual anchor is missing `_find_anchor_lines` will synthesise a combined row around the detected AOP code and tag it with `metadata["aop_fallback"] = True`; `_locate_numeric_cluster` must keep using the AOP column bounds in this mode so that split numeric tokens are preserved.
- Year columns (headers; locate X ranges by bbox):
  - `(?i)текућ[ае]\s+годин[ае]` → prefer by default
  - `(?i)претходн[ае]\s+годин[ае]`

> Keep **synonym lists** and minor spelling variants in `anchors.py` (e.g., extra spaces, all caps, mixed scripts).

### 2.4 Configuration (`config.yaml`)
```yaml
tesseract_path: "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
poppler_bin_dir: "C:\\tools\\poppler\\bin"     # contains pdftoppm/pdftocairo
ocr_langs: "srp+srp_latn+eng"
dpi: 300
multi_dpi: [300, 400]
confidence_threshold: 0.82
year_preference: "current"  # or "previous"
page_limit: 2               # first pages to OCR per PDF
overwrite_nonempty: false
use_pdfplumber_fallback: true
```

### 2.5 CLI (app.py)
```
Usage:
  app.exe --excel "<path-to-excel>" [--year current|previous] [--force]

Notes:
- PDFs are read from the **current working directory** (where app.exe resides).
- Default year is "current". Use "--year previous" to switch.
- Use "--force" to overwrite non-empty cells in G/H/I.
```
- Resolve Excel path: if relative, resolve against current directory.
- Validate presence of Tesseract/Poppler (fail-fast with clear error).

### 2.6 Caching & Performance
- Compute PDF file hash (e.g., SHA-1 of first megabytes + size + mtime) → cache OCR text/TSV + detected MB + form type.
- Store in SQLite (`cache.sqlite`) to avoid repeated OCR on unchanged files.
- Parallelize PDF OCR with a small worker pool (CPU-bound; be careful with Tesseract concurrency).

### 2.7 Logging & Reporting
- `logs/finstat_extractor.log` — rotating logs (INFO default; DEBUG via `--debug`).
- `report_missing.csv` columns: `MB;missing_bu;missing_bs;missing_fields;notes`.
- Persist structured diagnostics per extracted field (`source_file`, `page_idx`, OCR `method`, `confidence_avg`, raw/normalized value) either inside the log or a companion JSON/CSV for audit.
- Summaries at end: processed rows, filled values, missing counts, low-confidence retries triggered.

### 2.8 Edge Cases & Validation
- **Multiple PDFs** for same MB & form: choose latest by `mtime`; warn on duplicates.
- **Unreadable scans:** emit warning; suggest rescanning at 300 dpi grayscale.
- **Low-confidence OCR:** automatically trigger multi-DPI retries, two-pass numeric cropping, and pdfplumber fallback before declaring failure.
- **Non-numeric cells or misaligned picks:** extra numeric validation (max digits, not all zeros unless plausible).
- **Excel integrity:** back up original as `filename.bak.xlsx` (optional flag `--backup`).

### 2.9 Packaging (PyInstaller)
- Build single-file exe:
  - Ensure only **our Python code** is bundled.
  - Do **not** bundle Poppler or Tesseract; require local installation.
- `build.bat` example:
  - Sets `--add-data` for `config.yaml`.
  - Adds icon and version metadata.
  - Produces `dist/app.exe`.
- Post-build smoke test with sample PDFs and the provided Excel.

### 2.10 Security/Privacy
- No internet access required; all processing is local.
- No data exfiltration; logs avoid storing full numeric tables unless debugging is enabled.

### 2.11 Acceptance Criteria
- Given an Excel with valid **"Матични број"** values and a folder of mixed-form PDFs next to `app.exe`, running:
  - populates G/H/I correctly for ≥95% of legible forms (300 dpi).
  - generates `report_missing.csv` for the remainder with actionable reasons.
  - runs second time significantly faster due to cache.
  - does not overwrite non-empty cells unless `--force`.

---

## 3) Task List for ChatGPT Codex

1. **Bootstrap Project**
   - Create directory structure (see §2.1).
   - Initialize `config.yaml` with sensible defaults (OCR langs incl. `eng`, multi-DPI, confidence thresholds, pdfplumber fallback toggle).
   - Add `requirements.txt` (pytesseract, pdf2image, pillow, openpyxl, pandas, regex, python-dateutil, tqdm, click, loguru, sqlalchemy or sqlite3 stdlib, **opencv-python**, **numpy**, **pdfplumber**).

2. **Implement OCR Engine (`/ocr/ocr_engine.py`)**
   - Wrapper for Poppler (`pdftoppm` or `pdftocairo`) → PIL images (first N pages).
   - Tesseract calls to produce **plain text** and **TSV** per page, with support for two-pass OCR (anchor + numeric) and multi-DPI retries.
   - Integrate OpenCV preprocessing (grayscale, blur, adaptive threshold, deskew) before OCR.
   - Return structure: `{ text, tsv_rows (with bbox), page_map, diagnostics }` capturing confidence averages and methods used.
   - Fallback to `pdfplumber` text extraction when OCR confidence remains low and the PDF is text-based.
   - File-level caching keyed by hash.

3. **Anchors & Parsing (`/ocr/anchors.py`)**
   - Centralize regex for headers, MB, row labels, year headers.
   - Utilities to locate header column bboxes (current vs previous year).

4. **PDF Indexer (`/index/pdf_index.py`)**
   - Walk `cwd` for `*.pdf`.
   - For each: OCR; extract MB; detect form type (BU/BS); add to index map. Record per-file confidence diagnostics to support multi-DPI retries.
   - Resolve duplicates (prefer latest `mtime`); expose `get_pdf_for(MB, type)`.

5. **Extractors**
   - **BU (`/extract/bu.py`)**: find **"Пословни приходи"** anchor; pick nearest numeric token **in Current/Previous column** using TSV coordinates + column bbox detection.
   - **BS (`/extract/bs.py`)**: same for **"Укупна актива"** and **"Губитак изнад висине капитала"**; enforce `max(0, −AOP 0401)` logic.
   - Robust numeric normalization; return `Decimal` or `int` (thousands) plus diagnostics.

6. **Excel IO (`/io/excel_io.py`)**
   - Read Excel → list of rows with `MB` and row indices.
   - Ensure/create columns **G/H/I** with exact headers.
   - Write values; respect `overwrite_nonempty` flag; save workbook.
   - Persist diagnostic metadata alongside values (hidden columns or external sheet/report) capturing source file, page, method, and confidence.

7. **CLI Orchestration (`app.py`)**
   - Parse args (`--excel`, `--year`, `--force`, `--debug`).
   - Load config; set year preference dynamically.
   - Build PDF index from `cwd`.
   - Iterate Excel rows, extract values, write back, produce reports.
   - Emit summary; exit with non-zero code on fatal errors (missing Tesseract/Poppler).

8. **Packaging (`build.bat`)**
   - PyInstaller spec for single exe.
   - Document end-user steps in `README.md`.

9. **Smoke & Unit Tests**
   - Synthetic TSV lines to verify anchor→rightward-pick logic.
   - Golden-file tests for number normalization and Excel writing.

---

## 4) End-User Notes (to include in README)
1. Install **Tesseract OCR** (with language packs `srp`, `srp_latn`, `eng`) and **Poppler**; add both to `PATH` or set paths in `config.yaml`.
2. Place `app.exe`, the **Excel**, and all **PDFs** in the **same folder**.
3. Run from that folder:
   ```
   app.exe --excel "UPITNIK_primer 1.xls"
   ```
4. Check the Excel (columns G/H/I), `report_missing.csv`, and logs.

---

## 5) Non-Goals
- No GUI in the initial version (CLI only).
- No remote services / databases.
- No attempt to repair corrupted PDFs.

---

## 6) Risks & Mitigations
- **Low OCR quality** → Suggest 300 dpi scans; grayscale; deskew; allow `--page-limit` to extend pages.
- **Label drift** → Maintain synonyms; fuzzy matching with thresholds.
- **Table layout variance** → Rely on coordinates relative to row anchors + detected column headers.

---

**Ready for implementation.** This spec is intentionally concrete so the agent can scaffold code and ship a portable `app.exe` that colleagues can run locally with PDFs next to the executable.
