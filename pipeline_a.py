"""Pipeline A — Docling extraction with targeted LLM repair of bad tables.

Docling produces the base markdown. Each markdown table is scored with a set of
heuristics; tables that look broken are re-extracted from a rendered page image
with a vision LLM (gpt-4o-mini by default).
"""

from __future__ import annotations

import statistics
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from utils import (
    call_openai_vision,
    clean_markdown,
    count_tokens_cost,
    crop_image_region,
    image_to_base64,
    is_separator_row,
    pdf_page_to_image,
    split_table_row,
)

REPAIR_SYSTEM_PROMPT = """You are a precise document extraction assistant. Your task is to extract a table from a document image and reproduce it as a valid GitHub-Flavored Markdown (GFM) table.

Rules:
- Reproduce EVERY row and EVERY column exactly as they appear in the image.
- Do not skip rows, merge rows, or summarize content.
- For merged cells (cells that span multiple columns), repeat the content in each column.
- Numeric values must be reproduced exactly, including currency symbols, percentages, and decimal separators.
- If a cell contains a bullet list, represent it as semicolon-separated values within the cell.
- Output ONLY the markdown table. No preamble, no explanation, no code fences."""

REPAIR_USER_PROMPT = (
    "Extract the table visible in this image as a GitHub-Flavored Markdown table. "
    "The document is in Spanish. Reproduce all content exactly."
)


@dataclass
class TableCandidate:
    raw_markdown: str
    start_line: int
    end_line: int
    page_number: int  # 0-indexed
    quality_score: int = 0
    flagged: bool = False
    bbox_px: tuple | None = None  # pixel bbox at 200 DPI, if known
    issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Table detection in markdown
# ---------------------------------------------------------------------------
def _find_markdown_tables(md_lines: list[str]) -> list[TableCandidate]:
    """Locate contiguous blocks of ``|`` rows in the markdown."""
    candidates: list[TableCandidate] = []
    i = 0
    n = len(md_lines)
    while i < n:
        if md_lines[i].lstrip().startswith("|"):
            start = i
            while i < n and md_lines[i].lstrip().startswith("|"):
                i += 1
            end = i - 1
            # Need at least a header + separator to count as a table.
            block = md_lines[start : end + 1]
            if len(block) >= 2:
                candidates.append(
                    TableCandidate(
                        raw_markdown="\n".join(block),
                        start_line=start,
                        end_line=end,
                        page_number=0,
                    )
                )
        else:
            i += 1
    return candidates


# ---------------------------------------------------------------------------
# Docling provenance -> page number + pixel bbox
# ---------------------------------------------------------------------------
def _attach_docling_provenance(candidates, doc, dpi, logger) -> None:
    """Best-effort: map the i-th markdown table to the i-th docling table to
    pull its page number and bounding box. All failures degrade to page 0 /
    full-page crop."""
    try:
        tables = list(getattr(doc, "tables", []) or [])
    except Exception:  # noqa: BLE001
        tables = []

    scale = dpi / 72.0
    for idx, cand in enumerate(candidates):
        if idx >= len(tables):
            continue
        try:
            prov = tables[idx].prov[0]
            cand.page_number = max(0, int(prov.page_no) - 1)  # docling is 1-indexed
        except Exception:  # noqa: BLE001
            continue

        # Try to convert the bbox to top-left pixel coordinates.
        try:
            bbox = prov.bbox
            page = doc.pages[prov.page_no]
            page_height = float(page.size.height)
            # docling bbox uses a bottom-left origin; flip to top-left.
            try:
                tl = bbox.to_top_left_origin(page_height=page_height)
                x0, y0, x1, y1 = tl.l, tl.t, tl.r, tl.b
            except Exception:  # noqa: BLE001
                x0 = bbox.l
                x1 = bbox.r
                y0 = page_height - bbox.t
                y1 = page_height - bbox.b
            cand.bbox_px = (x0 * scale, y0 * scale, x1 * scale, y1 * scale)
        except Exception:  # noqa: BLE001
            cand.bbox_px = None


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------
def _score_table(cand: TableCandidate) -> None:
    lines = [ln for ln in cand.raw_markdown.splitlines() if ln.strip()]
    rows = [ln for ln in lines if not is_separator_row(ln)]
    if not rows:
        cand.quality_score = 99
        cand.flagged = True
        cand.issues = ["unparseable"]
        return

    parsed = [split_table_row(r) for r in rows]
    col_counts = [len(r) for r in parsed]
    data_rows = parsed[1:]  # everything after the header
    all_cells = [c for row in parsed for c in row]

    issues: list[str] = []

    # 1. Column consistency
    if len(col_counts) > 1 and statistics.pstdev(col_counts) > 0:
        issues.append("inconsistent_columns")

    # 2. Empty cell ratio
    if all_cells:
        empty = sum(1 for c in all_cells if c.strip() == "")
        if empty / len(all_cells) > 0.40:
            issues.append("high_empty_ratio")

    # 3. Truncation signal
    if any(c.rstrip().endswith("...") or c.rstrip().endswith("…") for c in all_cells):
        issues.append("truncation")

    # 4. Minimum rows
    if len(data_rows) < 2:
        issues.append("too_few_rows")

    # 5. Encoding artifacts
    artifacts = sum(cand.raw_markdown.count(ch) for ch in ("?", "□", "▪"))
    if artifacts > 3:
        issues.append("encoding_artifacts")

    # 6. Short cell content
    nonempty = [c for c in all_cells if c.strip()]
    if nonempty:
        avg_len = sum(len(c.strip()) for c in nonempty) / len(nonempty)
        if avg_len < 2:
            issues.append("short_cells")

    cand.issues = issues
    cand.quality_score = len(issues)
    cand.flagged = len(issues) >= 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_pipeline_a(pdf_path: Path, output_dir: Path, model: str, client, logger) -> dict:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pdf_path.stem}.md"

    result = {
        "pdf_name": pdf_path.name,
        "output_path": str(out_path),
        "total_cost": 0.0,
        "tables_found": 0,
        "tables_flagged": 0,
        "tables_repaired": 0,
    }

    # --- Step 1: Docling extraction --------------------------------------
    logger.info("Pipeline A: running docling on %s", pdf_path.name)
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    # Disable OCR for digital PDFs — avoids the torch.PP-OCRv6 init failure
    # seen when rapidocr-pytorch is installed without the matching model files.
    # Scanned PDFs would need do_ocr=True, but digital PDFs have embedded text.
    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_ocr = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
    )
    conv = converter.convert(str(pdf_path))
    doc = conv.document

    md = doc.export_to_markdown()
    try:
        num_pages = doc.num_pages()
    except Exception:  # noqa: BLE001
        num_pages = len(getattr(doc, "pages", {}) or {})
    logger.info("Docling detected %s page(s)", num_pages)

    md_lines = md.splitlines()

    # --- Step 2: Table detection ----------------------------------------
    candidates = _find_markdown_tables(md_lines)
    result["tables_found"] = len(candidates)
    logger.info("Found %d markdown table(s)", len(candidates))
    _attach_docling_provenance(candidates, doc, dpi=200, logger=logger)

    # --- Step 3: Quality scoring ----------------------------------------
    for idx, cand in enumerate(candidates):
        _score_table(cand)
        logger.debug(
            "Table %d (page %d): score=%d issues=%s flagged=%s",
            idx, cand.page_number, cand.quality_score, cand.issues, cand.flagged,
        )
        logger.info(
            "Table %d: quality_score=%d flagged=%s",
            idx, cand.quality_score, cand.flagged,
        )

    flagged = [c for c in candidates if c.flagged]
    result["tables_flagged"] = len(flagged)

    # --- Step 4: LLM repair of flagged tables ---------------------------
    # Process in reverse line order so earlier indices stay valid as we splice.
    for cand in sorted(flagged, key=lambda c: c.start_line, reverse=True):
        try:
            logger.info(
                "Repairing flagged table on page %d (issues: %s)",
                cand.page_number, cand.issues,
            )
            image = pdf_page_to_image(pdf_path, cand.page_number, dpi=200)
            if cand.bbox_px is not None:
                image = crop_image_region(image, cand.bbox_px)
                logger.debug("Cropped to bbox %s", cand.bbox_px)
            else:
                logger.debug("No bbox available; using full page")

            b64 = image_to_base64(image)
            text, usage = call_openai_vision(
                REPAIR_SYSTEM_PROMPT, REPAIR_USER_PROMPT, b64, model, client
            )
            cost = count_tokens_cost(usage, model)
            result["total_cost"] += cost
            logger.info(
                "Repair tokens=%s cost=$%.5f", usage.get("total_tokens"), cost
            )

            repaired = _strip_code_fences(text).strip()
            if repaired:
                new_lines = repaired.splitlines()
                md_lines[cand.start_line : cand.end_line + 1] = new_lines
                result["tables_repaired"] += 1
            else:
                logger.warning("Empty repair response; keeping original table")
        except Exception:  # noqa: BLE001
            logger.error("Table repair failed:\n%s", traceback.format_exc())

    # --- Step 5: Cleanup and save ---------------------------------------
    final_md = clean_markdown("\n".join(md_lines))
    out_path.write_text(final_md, encoding="utf-8")
    logger.info(
        "Pipeline A done: %d flagged, %d repaired -> %s",
        result["tables_flagged"], result["tables_repaired"], out_path,
    )
    return result


def _strip_code_fences(text: str) -> str:
    """Remove a leading/trailing ```...``` fence if the model added one."""
    lines = text.strip().splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)
