"""Pipeline C — PyMuPDF layout analysis + Camelot tables + LLM fallback.

Text and headings come from PyMuPDF's layout (fonts/sizes/positions). Tables are
extracted with Camelot (lattice, then stream); only when both fail does a vision
LLM (gpt-4o-mini) re-extract the cropped table region. Regions are finally
ordered top-to-bottom with simple two-column awareness.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from utils import (
    call_openai_vision,
    clean_markdown,
    count_tokens_cost,
    crop_image_region,
    image_to_base64,
)
from pipeline_a import REPAIR_SYSTEM_PROMPT, REPAIR_USER_PROMPT, _strip_code_fences

_BOLD_FLAG = 1 << 4  # PyMuPDF span flag bit for bold


@dataclass
class Region:
    page: int
    x0: float
    y0: float
    x1: float
    type: str  # "text" | "table"
    markdown: str = ""
    font_size: float = 0.0
    bold: bool = False
    text: str = ""
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _overlap_ratio(b: tuple, t: tuple) -> float:
    ix0, iy0 = max(b[0], t[0]), max(b[1], t[1])
    ix1, iy1 = min(b[2], t[2]), min(b[3], t[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    barea = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / barea


def _block_font(block: dict) -> tuple[float, bool, str]:
    """Return (max_font_size, is_bold, joined_text) for a text block."""
    max_size = 0.0
    bold = False
    lines_text: list[str] = []
    for line in block.get("lines", []):
        parts = []
        for span in line.get("spans", []):
            parts.append(span.get("text", ""))
            size = float(span.get("size", 0.0))
            if size > max_size:
                max_size = size
            flags = int(span.get("flags", 0))
            font = span.get("font", "") or ""
            if flags & _BOLD_FLAG or "bold" in font.lower():
                bold = True
        s = "".join(parts).strip()
        if s:
            lines_text.append(s)
    return max_size, bold, "\n".join(lines_text)


# ---------------------------------------------------------------------------
# Camelot -> markdown
# ---------------------------------------------------------------------------
def _md_cell(value) -> str:
    s = str(value).replace("\n", " ").strip()
    s = s.replace("|", "\\|")
    return s if s else " "


def camelot_to_markdown(table) -> str:
    """Convert a Camelot Table (via table.df) to a GFM markdown table."""
    df = table.df
    rows = df.values.tolist()
    if not rows:
        return ""
    header = [_md_cell(c) for c in rows[0]]
    ncols = len(header)
    md = ["| " + " | ".join(header) + " |"]
    md.append("| " + " | ".join(["---"] * ncols) + " |")
    for row in rows[1:]:
        cells = [_md_cell(c) for c in row]
        cells += [" "] * (ncols - len(cells))  # pad short rows
        md.append("| " + " | ".join(cells[:ncols]) + " |")
    return "\n".join(md)


def _consistent_columns(table) -> bool:
    df = table.df
    return df.shape[0] >= 1 and df.shape[1] >= 1


# ---------------------------------------------------------------------------
# Heading level assignment
# ---------------------------------------------------------------------------
def _build_heading_levels(heading_sizes: set[float]) -> dict[float, int]:
    """Map the largest heading font sizes to #, ##, ### (capped at ###)."""
    ordered = sorted(heading_sizes, reverse=True)
    levels: dict[float, int] = {}
    for idx, size in enumerate(ordered):
        levels[size] = min(idx + 1, 3)
    return levels


def _text_block_to_markdown(text: str, is_heading: bool, level: int) -> str:
    if is_heading:
        title = " ".join(text.split())
        return f"{'#' * level} {title}"

    # List detection: any line starting with a bullet marker.
    out_lines: list[str] = []
    is_list = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped[:1] in ("•", "○", "-", "·", "*"):
            is_list = True
            out_lines.append("- " + stripped[1:].strip())
        else:
            out_lines.append(stripped)
    if is_list:
        return "\n".join(out_lines)

    # Plain paragraph: join wrapped lines into one.
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Camelot extraction for a single table region
# ---------------------------------------------------------------------------
def _camelot_region(pdf_path, page_index, bbox, page_height, logger):
    """Try lattice then stream on the given region. Returns (md, mode) or
    (None, None) on failure."""
    import camelot

    x0, y0, x1, y1 = bbox
    # Camelot table_areas use a bottom-left origin: "left,top,right,bottom".
    area = f"{x0},{page_height - y0},{x1},{page_height - y1}"
    page_str = str(page_index + 1)

    # Attempt 1 — lattice
    try:
        tables = camelot.read_pdf(
            str(pdf_path), pages=page_str, flavor="lattice", table_areas=[area]
        )
        if tables.n >= 1:
            t = tables[0]
            acc = float(t.parsing_report.get("accuracy", 0))
            logger.debug("Lattice accuracy=%.1f", acc)
            if acc >= 80 and _consistent_columns(t):
                return camelot_to_markdown(t), "lattice"
    except Exception as exc:  # noqa: BLE001
        logger.debug("Lattice failed: %s", exc)

    # Attempt 2 — stream
    try:
        tables = camelot.read_pdf(
            str(pdf_path), pages=page_str, flavor="stream", table_areas=[area]
        )
        if tables.n >= 1:
            t = tables[0]
            acc = float(t.parsing_report.get("accuracy", 0))
            logger.debug("Stream accuracy=%.1f", acc)
            if acc >= 70 and _consistent_columns(t):
                return camelot_to_markdown(t), "stream"
    except Exception as exc:  # noqa: BLE001
        logger.debug("Stream failed: %s", exc)

    return None, None


# ---------------------------------------------------------------------------
# Reading-order merge
# ---------------------------------------------------------------------------
def _order_regions(regions: list[Region], page_widths: dict[int, float]) -> list[Region]:
    ordered: list[Region] = []
    by_page: dict[int, list[Region]] = {}
    for r in regions:
        by_page.setdefault(r.page, []).append(r)

    for page in sorted(by_page):
        page_regions = by_page[page]
        width = page_widths.get(page, 0.0) or 1.0
        x0s = [r.x0 for r in page_regions]
        multi_col = (max(x0s) - min(x0s)) > 0.40 * width if x0s else False
        midpoint = width / 2.0

        def sort_key(r: Region):
            col = 1 if (multi_col and r.x0 >= midpoint) else 0
            return (col, round(r.y0, 1), r.x0)

        ordered.extend(sorted(page_regions, key=sort_key))
    return ordered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_pipeline_c(pdf_path: Path, output_dir: Path, model: str, client, logger) -> dict:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pdf_path.stem}.md"

    result = {
        "pdf_name": pdf_path.name,
        "output_path": str(out_path),
        "total_cost": 0.0,
        "tables_camelot_success": 0,
        "tables_llm_fallback": 0,
        "tables_failed": 0,
    }

    regions: list[Region] = []
    page_widths: dict[int, float] = {}
    heading_sizes: set[float] = set()
    raw_text_blocks: list[dict] = []  # deferred until heading levels are known
    table_regions: list[dict] = []

    # --- Step 1 & 2: layout analysis + text extraction ------------------
    with fitz.open(pdf_path) as doc:
        logger.info("Pipeline C: %s has %d page(s)", pdf_path.name, doc.page_count)
        for p in range(doc.page_count):
            page = doc.load_page(p)
            page_widths[p] = float(page.rect.width)
            page_height = float(page.rect.height)

            # Table bounding boxes via find_tables (PyMuPDF >= 1.23).
            tbboxes: list[tuple] = []
            try:
                found = page.find_tables()
                for t in found.tables:
                    tbboxes.append(tuple(float(v) for v in t.bbox))
            except Exception as exc:  # noqa: BLE001
                logger.debug("find_tables unavailable on page %d: %s", p + 1, exc)

            for tb in tbboxes:
                table_regions.append(
                    {"page": p, "bbox": tb, "page_height": page_height}
                )
            logger.info("Page %d: %d table region(s) detected", p + 1, len(tbboxes))

            # Text blocks not overlapping any table.
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                if block.get("type", 0) != 0:  # 0 == text block
                    continue
                bbox = tuple(float(v) for v in block["bbox"])
                if any(_overlap_ratio(bbox, tb) > 0.5 for tb in tbboxes):
                    continue
                size, bold, text = _block_font(block)
                if not text.strip():
                    continue
                short = len(text.replace("\n", " ")) < 80
                is_heading = size > 16 or (bold and short)
                if is_heading:
                    heading_sizes.add(round(size, 1))
                raw_text_blocks.append(
                    {
                        "page": p,
                        "bbox": bbox,
                        "size": round(size, 1),
                        "bold": bold,
                        "text": text,
                        "is_heading": is_heading,
                    }
                )

    # --- Heading levels (global ranking) --------------------------------
    heading_levels = _build_heading_levels(heading_sizes)

    for blk in raw_text_blocks:
        level = heading_levels.get(blk["size"], 3) if blk["is_heading"] else 0
        md = _text_block_to_markdown(blk["text"], blk["is_heading"], level)
        regions.append(
            Region(
                page=blk["page"],
                x0=blk["bbox"][0],
                y0=blk["bbox"][1],
                x1=blk["bbox"][2],
                type="text",
                markdown=md,
                font_size=blk["size"],
                bold=blk["bold"],
                text=blk["text"],
                bbox=blk["bbox"],
            )
        )

    # --- Step 3: table extraction ---------------------------------------
    for tr in table_regions:
        p = tr["page"]
        bbox = tr["bbox"]
        page_height = tr["page_height"]
        md = None
        try:
            md, mode = _camelot_region(pdf_path, p, bbox, page_height, logger)
        except Exception:  # noqa: BLE001
            logger.error("Camelot crashed:\n%s", traceback.format_exc())
            md, mode = None, None

        if md:
            result["tables_camelot_success"] += 1
            logger.info("Table on page %d extracted via Camelot (%s)", p + 1, mode)
        else:
            # Attempt 3 — LLM fallback
            try:
                logger.info("Camelot failed on page %d; LLM fallback", p + 1)
                dpi = 200
                scale = dpi / 72.0
                image = _render_page_image(pdf_path, p, dpi)
                px_bbox = (
                    bbox[0] * scale,
                    bbox[1] * scale,
                    bbox[2] * scale,
                    bbox[3] * scale,
                )
                crop = crop_image_region(image, px_bbox)
                b64 = image_to_base64(crop)
                text, usage = call_openai_vision(
                    REPAIR_SYSTEM_PROMPT, REPAIR_USER_PROMPT, b64, model, client
                )
                cost = count_tokens_cost(usage, model)
                result["total_cost"] += cost
                logger.info(
                    "Fallback tokens=%s cost=$%.5f", usage.get("total_tokens"), cost
                )
                md = _strip_code_fences(text).strip()
                if md:
                    result["tables_llm_fallback"] += 1
                else:
                    result["tables_failed"] += 1
                    logger.warning("Empty LLM fallback on page %d", p + 1)
            except Exception:  # noqa: BLE001
                result["tables_failed"] += 1
                logger.error("LLM fallback failed:\n%s", traceback.format_exc())
                md = None

        if md:
            regions.append(
                Region(
                    page=p,
                    x0=bbox[0],
                    y0=bbox[1],
                    x1=bbox[2],
                    type="table",
                    markdown=md,
                    bbox=bbox,
                )
            )

    # --- Step 4: reading-order merge ------------------------------------
    ordered = _order_regions(regions, page_widths)

    # --- Step 5: cleanup and save ---------------------------------------
    body = "\n\n".join(r.markdown for r in ordered if r.markdown.strip())
    final_md = clean_markdown(body)
    out_path.write_text(final_md, encoding="utf-8")
    logger.info(
        "Pipeline C done: camelot=%d fallback=%d failed=%d -> %s",
        result["tables_camelot_success"],
        result["tables_llm_fallback"],
        result["tables_failed"],
        out_path,
    )
    return result


def _render_page_image(pdf_path, page_index, dpi) -> Image.Image:
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
