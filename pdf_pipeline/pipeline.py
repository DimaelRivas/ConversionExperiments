"""Core pipeline: Docling → PyMuPDF structural validation → LLM repair.

Docling is the primary extractor for all content. PyMuPDF's find_tables() is
used only as a geometric reality-check — it never extracts cell content. The LLM
is called exclusively on tightly cropped images of individual tables that fail
validation. Full-page images are never sent to the LLM.
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from diff_logger import log_table_repair, write_repair_summary
from repairer import calculate_cost, repair_table
from validator import (
    TableStructure,
    get_page_table_structures,
    parse_docling_table_structure,
    validate_table,
)


@dataclass
class TableCandidate:
    raw_markdown: str
    start_line: int
    end_line: int
    page_number: int           # 0-indexed
    table_index_on_page: int   # 0-indexed among tables on this page
    bbox: tuple | None         # (x0,y0,x1,y1) PDF points, PyMuPDF top-left origin
    context_before: str
    context_after: str
    validated: bool = False
    repaired: bool = False
    repair_cost: float = 0.0
    is_orphan: bool = False
    repair_reason: str = ""
    repaired_markdown: str = ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    pdf_path: Path,
    output_dir: Path,
    model: str,
    client,
    logger: logging.Logger,
    repairs_dir: Path | None = None,
) -> dict:
    """Run the full pipeline on one PDF. Returns a structured result dict."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pdf_path.stem}.md"

    # Derive repairs dir from output dir if not given explicitly
    if repairs_dir is None:
        repairs_dir = output_dir.parent / "logs" / "repairs"
    repairs_dir = Path(repairs_dir)
    repairs_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "pdf_name":        pdf_path.name,
        "output_path":     out_path,
        "total_cost":      0.0,
        "tables_found":    0,
        "tables_validated":0,
        "tables_repaired": 0,
        "tables_orphaned": 0,
        "pages_processed": 0,
        "repairs":         [],          # [{pdf_name, page_number, table_index, ...}]
    }

    # -----------------------------------------------------------------------
    # Step 1 — Docling extraction
    # -----------------------------------------------------------------------
    logger.info("Step 1: Docling extraction — %s", pdf_path.name)
    t0 = time.monotonic()

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_ocr = False   # Digital PDFs have embedded text; OCR init crashes otherwise

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
    )
    conv = converter.convert(str(pdf_path))
    doc = conv.document

    md: str = doc.export_to_markdown()
    doc_dict: dict = _export_docling_dict(doc)

    try:
        pages_count = doc.num_pages()
    except Exception:
        pages_count = len(doc_dict.get("pages") or {})
    result["pages_processed"] = pages_count

    json_tables: list[dict] = list(doc_dict.get("tables") or [])
    elapsed = time.monotonic() - t0
    logger.info(
        "Docling done: %d pages, %d JSON table(s), %.1fs",
        pages_count, len(json_tables), elapsed,
    )

    # -----------------------------------------------------------------------
    # Step 1b — Build TableCandidate list from markdown + JSON metadata
    # -----------------------------------------------------------------------
    md_lines = md.splitlines()
    table_blocks = _find_table_blocks(md_lines)
    result["tables_found"] = len(table_blocks)
    logger.info("Parsed %d markdown table block(s)", len(table_blocks))

    if len(table_blocks) != len(json_tables):
        logger.warning(
            "Count mismatch: %d markdown tables vs %d JSON tables — "
            "some table metadata may be unavailable",
            len(table_blocks), len(json_tables),
        )

    page_table_counter: Counter[int] = Counter()
    candidates: list[TableCandidate] = []

    for i, (start_line, end_line, raw_md) in enumerate(table_blocks):
        page_number = 0
        bbox: tuple | None = None

        if i < len(json_tables):
            json_tbl = json_tables[i]
            prov_list = list(json_tbl.get("prov") or [])
            if prov_list:
                prov = prov_list[0]
                page_no_1idx = int(prov.get("page_no") or 1)
                page_number = page_no_1idx - 1   # 0-indexed

                page_h = _get_page_height(doc_dict, page_no_1idx)
                bbox_data = prov.get("bbox")
                if bbox_data:
                    bbox = _convert_docling_bbox(bbox_data, page_h)

        table_idx_on_page = page_table_counter[page_number]
        page_table_counter[page_number] += 1

        # Gather surrounding text as context for the LLM
        before_raw = [
            ln.strip()
            for ln in md_lines[max(0, start_line - 6) : start_line]
            if ln.strip() and not ln.strip().startswith("|")
        ]
        after_raw = [
            ln.strip()
            for ln in md_lines[end_line + 1 : end_line + 7]
            if ln.strip() and not ln.strip().startswith("|")
        ]
        context_before = " ".join(before_raw)[:200]
        context_after  = " ".join(after_raw)[:200]

        candidates.append(
            TableCandidate(
                raw_markdown=raw_md,
                start_line=start_line,
                end_line=end_line,
                page_number=page_number,
                table_index_on_page=table_idx_on_page,
                bbox=bbox,
                context_before=context_before,
                context_after=context_after,
            )
        )

    # -----------------------------------------------------------------------
    # Step 2 — Structural validation via PyMuPDF
    # -----------------------------------------------------------------------
    logger.info("Step 2: Validating %d table(s) against PyMuPDF geometry", len(candidates))

    # Group by page so we make one find_tables() call per page
    by_page: dict[int, list[TableCandidate]] = {}
    for cand in candidates:
        by_page.setdefault(cand.page_number, []).append(cand)

    # Cache PyMuPDF structures; needed again in Step 3
    pymupdf_cache: dict[int, list[TableStructure]] = {}
    flagged: list[TableCandidate] = []

    for page_num in sorted(by_page):
        page_cands = by_page[page_num]
        pymupdf_structs = get_page_table_structures(pdf_path, page_num)
        pymupdf_cache[page_num] = pymupdf_structs

        valid_on_page = 0
        for cand in page_cands:
            doc_cols, doc_rows = parse_docling_table_structure(cand.raw_markdown)

            if cand.table_index_on_page < len(pymupdf_structs):
                pymupdf_struct = pymupdf_structs[cand.table_index_on_page]
            elif pymupdf_structs:
                pymupdf_struct = pymupdf_structs[-1]
            else:
                # PyMuPDF found no tables on this page — trust docling
                cand.validated = True
                valid_on_page += 1
                logger.debug(
                    "Page %d: PyMuPDF found 0 tables; trusting docling for table %d",
                    page_num + 1, cand.table_index_on_page,
                )
                continue

            is_valid, reason = validate_table(doc_cols, doc_rows, pymupdf_struct)
            if is_valid:
                cand.validated = True
                valid_on_page += 1
            else:
                cand.repair_reason = reason
                # Use PyMuPDF bbox when docling didn't provide one
                if cand.bbox is None:
                    cand.bbox = pymupdf_struct.bbox
                flagged.append(cand)
                logger.info("FLAGGED — %s", reason)

        logger.info(
            "Page %d: %d docling / %d pymupdf / %d valid / %d flagged",
            page_num + 1, len(page_cands), len(pymupdf_structs),
            valid_on_page, len(page_cands) - valid_on_page,
        )

        # Orphan detection: tables PyMuPDF found but docling didn't
        orphan_count = max(0, len(pymupdf_structs) - len(page_cands))
        if orphan_count > 0:
            logger.info(
                "Page %d: %d orphan table(s) (found by PyMuPDF, missed by docling)",
                page_num + 1, orphan_count,
            )
            result["tables_orphaned"] += orphan_count
            last_end = max((c.end_line for c in page_cands), default=len(md_lines) - 1)
            for j in range(len(page_cands), len(pymupdf_structs)):
                orphan_struct = pymupdf_structs[j]
                orphan_cand = TableCandidate(
                    raw_markdown="",
                    start_line=last_end + 1,
                    end_line=last_end,
                    page_number=page_num,
                    table_index_on_page=j,
                    bbox=orphan_struct.bbox,
                    context_before="",
                    context_after="",
                    is_orphan=True,
                    repair_reason=(
                        f"Orphan: table detected by PyMuPDF (col={orphan_struct.col_count}, "
                        f"row={orphan_struct.row_count}) but not extracted by docling"
                    ),
                )
                flagged.append(orphan_cand)
                candidates.append(orphan_cand)

    # Check pages with zero docling tables where PyMuPDF still finds tables
    all_page_nums = set(range(pages_count))
    for page_num in sorted(all_page_nums - set(by_page.keys())):
        pymupdf_structs = get_page_table_structures(pdf_path, page_num)
        if not pymupdf_structs:
            continue
        pymupdf_cache[page_num] = pymupdf_structs
        logger.info(
            "Page %d: 0 docling tables, %d PyMuPDF tables (all orphans)",
            page_num + 1, len(pymupdf_structs),
        )
        result["tables_orphaned"] += len(pymupdf_structs)
        for j, struct in enumerate(pymupdf_structs):
            orphan_cand = TableCandidate(
                raw_markdown="",
                start_line=len(md_lines),
                end_line=len(md_lines) - 1,
                page_number=page_num,
                table_index_on_page=j,
                bbox=struct.bbox,
                context_before="",
                context_after="",
                is_orphan=True,
                repair_reason=(
                    f"Orphan on page with no docling tables "
                    f"(pymupdf col={struct.col_count}, row={struct.row_count})"
                ),
            )
            flagged.append(orphan_cand)
            candidates.append(orphan_cand)

    result["tables_validated"] = sum(1 for c in candidates if c.validated and not c.is_orphan)

    logger.info(
        "Validation done: %d valid, %d flagged (%d orphans)",
        result["tables_validated"], len(flagged), result["tables_orphaned"],
    )

    # -----------------------------------------------------------------------
    # Step 3 — LLM repair
    # -----------------------------------------------------------------------
    if not flagged:
        logger.info("Step 3: No tables flagged — skipping LLM repair")
    else:
        logger.info("Step 3: Repairing %d table(s) via LLM", len(flagged))

    non_orphans = sorted(
        (c for c in flagged if not c.is_orphan),
        key=lambda c: c.start_line,
        reverse=True,   # process in reverse so line-index replacements don't shift later entries
    )
    orphans = [c for c in flagged if c.is_orphan]

    for cand in non_orphans + orphans:
        logger.info(
            "Repairing page %d table %d: %s",
            cand.page_number + 1, cand.table_index_on_page + 1, cand.repair_reason,
        )
        context = (cand.context_before + "\n" + cand.context_after).strip()
        repaired_md, usage_dict = repair_table(
            pdf_path, cand.page_number, cand.bbox, model, client, context
        )
        cost = calculate_cost(usage_dict, model)
        result["total_cost"] += cost
        cand.repair_cost = cost

        if repaired_md:
            cand.repaired = True
            cand.repaired_markdown = repaired_md
            result["tables_repaired"] += 1
            logger.info(
                "Repair OK — page %d table %d tokens=%d cost=$%.5f",
                cand.page_number + 1, cand.table_index_on_page + 1,
                usage_dict.get("total_tokens", 0), cost,
            )
        else:
            logger.warning(
                "Repair returned empty — page %d table %d",
                cand.page_number + 1, cand.table_index_on_page + 1,
            )

        doc_cols, doc_rows = parse_docling_table_structure(cand.raw_markdown)
        rep_cols, _       = (
            parse_docling_table_structure(repaired_md) if repaired_md else (0, 0)
        )
        pymupdf_structs = pymupdf_cache.get(cand.page_number, [])
        pymupdf_struct: TableStructure | None = (
            pymupdf_structs[cand.table_index_on_page]
            if cand.table_index_on_page < len(pymupdf_structs)
            else None
        )

        log_table_repair(
            pdf_name=pdf_path.name,
            page_number=cand.page_number,
            table_index=cand.table_index_on_page,
            reason=cand.repair_reason,
            original_markdown=cand.raw_markdown,
            repaired_markdown=repaired_md,
            log_dir=repairs_dir,
            usage_dict=usage_dict,
            cost=cost,
            docling_cols=doc_cols,
            docling_rows=doc_rows,
            pymupdf_cols=pymupdf_struct.col_count if pymupdf_struct else 0,
            pymupdf_rows=pymupdf_struct.row_count if pymupdf_struct else 0,
        )

        result["repairs"].append(
            {
                "pdf_name":    pdf_path.name,
                "page_number": cand.page_number,
                "table_index": cand.table_index_on_page,
                "reason":      cand.repair_reason,
                "cols_before": doc_cols,
                "cols_after":  rep_cols if repaired_md else doc_cols,
                "cost":        cost,
            }
        )

    # Write repair summary
    if result["repairs"]:
        write_repair_summary(result["repairs"], repairs_dir)

    # -----------------------------------------------------------------------
    # Step 4 — Document assembly
    # -----------------------------------------------------------------------
    logger.info("Step 4: Assembling final markdown")

    out_lines = list(md_lines)

    # Apply non-orphan repairs in reverse line order
    for cand in non_orphans:
        if not cand.repaired:
            continue
        replacement = cand.repaired_markdown.strip().splitlines()
        out_lines[cand.start_line : cand.end_line + 1] = replacement

    # Append orphan repairs at end of document
    for cand in orphans:
        if not cand.repaired:
            continue
        out_lines.extend(["", ""])
        out_lines.extend(cand.repaired_markdown.strip().splitlines())

    final_md = clean_markdown("\n".join(out_lines))

    # -----------------------------------------------------------------------
    # Step 5 — Write output
    # -----------------------------------------------------------------------
    out_path.write_text(final_md, encoding="utf-8")
    logger.info(
        "Written: %s (%d bytes)",
        out_path,
        len(final_md.encode("utf-8")),
    )

    return result


# ---------------------------------------------------------------------------
# Docling JSON helpers
# ---------------------------------------------------------------------------

def _export_docling_dict(doc) -> dict:
    """Serialize a docling DoclingDocument to a plain dict.

    Tries docling's own method first, then pydantic v2, then v1 — handles
    different docling/pydantic version combinations without crashing.
    """
    for method_name in ("export_to_dict",):
        method = getattr(doc, method_name, None)
        if method is not None:
            try:
                result = method()
                if result:
                    return dict(result)
            except Exception:
                pass

    for method_name in ("model_dump_json",):
        method = getattr(doc, method_name, None)
        if method is not None:
            try:
                raw = method()
                if isinstance(raw, bytes):
                    raw = raw.decode()
                return json.loads(raw) or {}
            except Exception:
                pass

    for method_name in ("model_dump", "dict"):
        method = getattr(doc, method_name, None)
        if method is not None:
            try:
                result = method()
                if result:
                    return dict(result)
            except Exception:
                pass

    return {}


def _get_page_height(doc_dict: dict, page_no_1idx: int) -> float:
    """Return page height in PDF points from docling's JSON. page_no_1idx is 1-indexed.

    Handles the two forms seen in practice:
    - dict keyed by str(page_no)  e.g. {"1": {"size": {"height": 792}}}
    - list indexed by 0-based position
    Defaults to 792 (US Letter) if not found.
    """
    pages = doc_dict.get("pages") or {}
    page_data: dict | None = None

    if isinstance(pages, dict):
        page_data = pages.get(str(page_no_1idx)) or pages.get(page_no_1idx)
    elif isinstance(pages, list) and len(pages) >= page_no_1idx:
        page_data = pages[page_no_1idx - 1]

    if page_data:
        size = page_data.get("size") or {}
        h = size.get("height")
        if h is not None:
            return float(h)

    return 792.0


def _convert_docling_bbox(bbox_data: dict, page_height: float) -> tuple:
    """Convert a docling provenance bbox to PyMuPDF top-left coordinates.

    Docling bbox fields (regardless of version):
        l  – left x
        t  – top  y  in the coord_origin system
        r  – right x
        b  – bottom y in the coord_origin system

    When coord_origin == BOTTOMLEFT (the PDF default):
        t > b  (t is the visually upper edge, with larger PDF y)
        Conversion to screen / PyMuPDF (top-left origin, y↓):
            x0 = l
            y0 = page_height - t   ← visually top of box, small screen y
            x1 = r
            y1 = page_height - b   ← visually bottom, large screen y
        Result: y0 < y1  ✓

    When coord_origin == TOPLEFT: fields are already in screen coords.
    """
    l = float(bbox_data.get("l") or 0)
    t = float(bbox_data.get("t") or 0)
    r = float(bbox_data.get("r") or 0)
    b = float(bbox_data.get("b") or 0)
    origin = str(bbox_data.get("coord_origin") or "BOTTOMLEFT").upper()

    if origin == "BOTTOMLEFT":
        return (l, page_height - t, r, page_height - b)
    else:
        return (l, t, r, b)


# ---------------------------------------------------------------------------
# Markdown utilities
# ---------------------------------------------------------------------------

def _find_table_blocks(md_lines: list[str]) -> list[tuple[int, int, str]]:
    """Find contiguous runs of | rows. Returns [(start, end, raw_md), ...]."""
    blocks: list[tuple[int, int, str]] = []
    i = 0
    n = len(md_lines)
    while i < n:
        if md_lines[i].lstrip().startswith("|"):
            start = i
            while i < n and md_lines[i].lstrip().startswith("|"):
                i += 1
            end = i - 1
            if end > start:  # need at least 2 rows (header + separator or data)
                raw_md = "\n".join(md_lines[start : end + 1])
                blocks.append((start, end, raw_md))
        else:
            i += 1
    return blocks


def clean_markdown(text: str) -> str:
    """Post-process assembled markdown.

    - Remove lines that appear ≥ 3 times identically (page headers/footers)
      except table rows (lines starting with |).
    - Collapse 3+ consecutive blank lines to 2.
    - Strip trailing whitespace from every line.
    - Ensure every markdown table has a blank line before and after.
    """
    if not text:
        return ""

    lines = text.splitlines()
    freq: Counter[str] = Counter(
        ln.strip()
        for ln in lines
        if ln.strip() and not ln.strip().startswith("|")
    )
    repeated = {s for s, n in freq.items() if n >= 3}

    # Remove repeated header/footer lines
    kept = [ln.rstrip() for ln in lines if ln.strip() not in repeated]

    # Collapse blank-line runs
    squashed: list[str] = []
    blank_run = 0
    for ln in kept:
        if ln == "":
            blank_run += 1
            if blank_run <= 2:
                squashed.append(ln)
        else:
            blank_run = 0
            squashed.append(ln)

    # Ensure blank lines around tables
    result: list[str] = []
    n = len(squashed)
    for i, ln in enumerate(squashed):
        is_table = ln.lstrip().startswith("|")
        prev_is_table = result[-1].lstrip().startswith("|") if result else False

        if is_table and result and not prev_is_table and result[-1] != "":
            result.append("")

        result.append(ln)

        if is_table:
            nxt = squashed[i + 1] if i + 1 < n else ""
            if nxt and not nxt.lstrip().startswith("|"):
                result.append("")

    return "\n".join(result).strip() + "\n"
