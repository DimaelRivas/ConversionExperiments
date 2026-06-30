"""Core pipeline: Docling extraction → markdown scan → review markers.

No LLM, no Camelot, no PyMuPDF, no network calls (after docling's one-time model
download). Docling's text/heading/list/table output is preserved verbatim; the
only modification is the insertion of HTML-comment review markers around tables
that the markdown scanner flags as structurally suspect.

The flagging decision is 100% markdown-based (see scanner.py). Page numbers shown
in the markers are read from docling's own JSON export (not from any PDF geometry
library) purely to give the reviewer a place to look.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path

from checklist import generate_pdf_checklist
from marker import apply_all_markers
from scanner import FlaggedTable, parse_tables, scan_markdown


def run_pipeline(
    pdf_path: Path,
    output_dir: Path,
    checklist_dir: Path,
    logger: logging.Logger,
) -> dict:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    checklist_dir = Path(checklist_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checklist_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pdf_path.stem}.md"

    # -----------------------------------------------------------------------
    # Step 1 — Docling extraction
    # -----------------------------------------------------------------------
    logger.info("Step 1: Docling extraction — %s", pdf_path.name)
    t0 = time.monotonic()

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_ocr = False  # digital PDFs carry embedded text; OCR init can crash

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
    )
    conv = converter.convert(str(pdf_path))
    doc = conv.document

    docling_markdown = doc.export_to_markdown()
    doc_dict = _export_docling_dict(doc)

    try:
        pages_count = doc.num_pages()
    except Exception:
        pages_count = len(doc_dict.get("pages") or {})

    lines = docling_markdown.splitlines()
    all_tables = parse_tables(lines)
    json_tables = list(doc_dict.get("tables") or [])
    logger.info(
        "Docling done: %d pages, %d markdown table(s), %.1fs",
        pages_count, len(all_tables), time.monotonic() - t0,
    )

    # Map each parsed table -> (page_number, table_index_on_page, section_title)
    # using docling's JSON (document order matches markdown order).
    page_counter: Counter[int] = Counter()
    meta_by_start: dict[int, dict] = {}
    for i, t in enumerate(all_tables):
        page_number = None
        if i < len(json_tables):
            prov = json_tables[i].get("prov") or []
            if prov and prov[0].get("page_no") is not None:
                page_number = int(prov[0]["page_no"]) - 1  # docling JSON is 1-indexed

        key = page_number if page_number is not None else -1
        table_idx = page_counter[key]
        page_counter[key] += 1

        meta_by_start[t["start_line"]] = {
            "page_number": page_number,
            "table_index_on_page": table_idx,
            "section_title": _nearest_heading(lines, t["start_line"]),
        }

    # -----------------------------------------------------------------------
    # Step 2 — Markdown scanning
    # -----------------------------------------------------------------------
    flagged_tables: list[FlaggedTable] = scan_markdown(docling_markdown)
    logger.info("%d tables flagged for review", len(flagged_tables))

    # -----------------------------------------------------------------------
    # Step 3 — Build flagged lists and insert markers
    # -----------------------------------------------------------------------
    flagged_for_markers: list[dict] = []
    flagged_details: list[dict] = []
    for f in flagged_tables:
        meta = meta_by_start.get(f.start_line, {})
        entry = {
            "start_line": f.start_line,
            "end_line": f.end_line,
            "reason": f.reason,
            "heuristic": f.heuristic,
            "page_number": meta.get("page_number"),
            "table_index_on_page": meta.get("table_index_on_page"),
            "section_title": meta.get("section_title", ""),
            "col_count": f.col_count,
        }
        flagged_for_markers.append(entry)
        flagged_details.append(entry)

    marked_md = apply_all_markers(docling_markdown, flagged_for_markers, pdf_path.name)

    # -----------------------------------------------------------------------
    # Step 4 — Clean and write output
    # -----------------------------------------------------------------------
    logger.info("Step 4: Cleaning markdown (markers preserved)")
    final_md = clean_markdown(marked_md)
    out_path.write_text(final_md, encoding="utf-8")
    logger.info("Written: %s (%d bytes)", out_path, len(final_md.encode("utf-8")))

    # -----------------------------------------------------------------------
    # Step 5 — Per-PDF checklist
    # -----------------------------------------------------------------------
    distinct_flagged = {f.start_line for f in flagged_tables}
    result = {
        "pdf_name": pdf_path.name,
        "output_path": out_path,
        "pages_processed": pages_count,
        "tables_found": len(all_tables),
        "tables_flagged": len(distinct_flagged),
        "tables_flagged_image": sum(1 for f in flagged_tables if f.heuristic == "image_proximity"),
        "tables_flagged_split": sum(1 for f in flagged_tables if f.heuristic == "split_table"),
        "tables_clean": len(all_tables) - len(distinct_flagged),
        "flagged_details": flagged_details,
    }

    generate_pdf_checklist(
        pdf_path.name,
        flagged_details,
        checklist_dir,
        total_tables=result["tables_found"],
        total_pages=pages_count,
    )
    logger.info("Checklist written for %s", pdf_path.name)

    return result


# ---------------------------------------------------------------------------
# Docling JSON helpers
# ---------------------------------------------------------------------------

def _export_docling_dict(doc) -> dict:
    """Serialize a DoclingDocument to a plain dict across docling/pydantic versions."""
    method = getattr(doc, "export_to_dict", None)
    if method is not None:
        try:
            r = method()
            if r:
                return dict(r)
        except Exception:
            pass
    method = getattr(doc, "model_dump_json", None)
    if method is not None:
        try:
            raw = method()
            if isinstance(raw, bytes):
                raw = raw.decode()
            return json.loads(raw) or {}
        except Exception:
            pass
    for name in ("model_dump", "dict"):
        method = getattr(doc, name, None)
        if method is not None:
            try:
                r = method()
                if r:
                    return dict(r)
            except Exception:
                pass
    return {}


# ---------------------------------------------------------------------------
# Markdown utilities
# ---------------------------------------------------------------------------

def _nearest_heading(md_lines: list[str], start_line: int) -> str:
    """Return the text of the nearest markdown heading above start_line, if any."""
    for j in range(start_line - 1, max(-1, start_line - 40), -1):
        if j < 0:
            break
        s = md_lines[j].strip()
        if s.startswith("#"):
            return s.lstrip("#").strip()
    return ""


def clean_markdown(text: str) -> str:
    """Clean assembled markdown WITHOUT touching HTML-comment markers.

    - Remove lines appearing ≥ 3 times identically (page headers/footers),
      never removing comment lines or table rows.
    - Collapse 3+ consecutive blank lines to 2.
    - Strip trailing whitespace per line.
    """
    if not text:
        return ""

    lines = text.splitlines()

    # Mark which lines are inside HTML comments (inclusive of the <!-- and --> lines)
    protected = [False] * len(lines)
    in_comment = False
    for idx, ln in enumerate(lines):
        opens = "<!--" in ln
        closes = "-->" in ln
        if in_comment:
            protected[idx] = True
            if closes:
                in_comment = False
            continue
        if opens:
            protected[idx] = True
            if not closes:  # multi-line comment continues
                in_comment = True
            continue

    # Frequency of plain text lines (exclude comments, tables, blanks)
    freq: Counter[str] = Counter()
    for idx, ln in enumerate(lines):
        s = ln.strip()
        if not s or protected[idx] or s.startswith("|"):
            continue
        freq[s] += 1
    repeated = {s for s, n in freq.items() if n >= 3}

    kept: list[str] = []
    for idx, ln in enumerate(lines):
        if not protected[idx] and ln.strip() in repeated and not ln.lstrip().startswith("|"):
            continue
        kept.append(ln.rstrip())

    # Collapse blank runs to a maximum of two
    out: list[str] = []
    blank_run = 0
    for ln in kept:
        if ln == "":
            blank_run += 1
            if blank_run <= 2:
                out.append(ln)
        else:
            blank_run = 0
            out.append(ln)

    return "\n".join(out).strip() + "\n"
