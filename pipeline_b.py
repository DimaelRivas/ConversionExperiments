"""Pipeline B — Full-page vision LLM extraction (the reference / ground truth).

Every page is rendered to an image and converted to markdown by a high-quality
vision model (gpt-4o by default). A regex-only continuity pass then stitches
paragraphs and tables that were split across page boundaries.
"""

from __future__ import annotations

import re
import traceback
from pathlib import Path

import fitz  # PyMuPDF

from utils import (
    call_openai_vision,
    clean_markdown,
    count_tokens_cost,
    image_to_base64,
    is_separator_row,
)
from PIL import Image

PAGE_SYSTEM_PROMPT = """You are a precise document-to-markdown conversion assistant. Convert the content of this document page to clean GitHub-Flavored Markdown (GFM).

Rules:
- Use # for the document's main title, ## for section headings, ### for subsections. Infer heading levels from font size and visual weight.
- Reproduce ALL tables as valid GFM markdown tables. Every row and column must be present. Do not skip or summarize rows.
- For tables with color-coded rows or merged cells, reproduce the content faithfully. For merged cells, repeat the value in each affected column.
- Reproduce bullet lists as markdown lists with `-`.
- Reproduce numbered lists as `1.`, `2.`, etc.
- Ignore page headers, page footers, and page numbers — do not include them in the output.
- For pie charts or bar charts, produce a markdown table summarizing the data labels and values visible in the chart. Add a one-line italic description above the table: _[Chart: description]_.
- For images or logos, write: `[IMAGE: brief description]`
- Preserve bold text using **bold** where visually prominent in the source.
- The document is in Spanish. Preserve all Spanish text exactly — do not translate.
- Output ONLY the markdown content. No preamble, no explanation, no code fences."""

PAGE_USER_PROMPT = "Convert this document page to markdown."


def _render_page(doc, page_index: int, dpi: int = 150) -> Image.Image:
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _strip_code_fences(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def _repair_continuity(pages: list[str]) -> str:
    """String/regex-only join of paragraphs and tables split across pages."""
    if not pages:
        return ""

    merged = pages[0].rstrip("\n")
    for nxt in pages[1:]:
        nxt = nxt.lstrip("\n")
        prev_lines = merged.splitlines()
        next_lines = nxt.splitlines()
        if not prev_lines or not next_lines:
            merged = (merged + "\n\n" + nxt).strip()
            continue

        last = prev_lines[-1].rstrip()
        first = next_lines[0].lstrip()

        # Case 1: a table continues across the break. Last line of page N is a
        # data row, first line of page N+1 is a `|` row that is NOT a header
        # separator -> concatenate directly (no blank line between).
        if (
            last.startswith("|")
            and first.startswith("|")
            and not is_separator_row(first)
        ):
            merged = "\n".join(prev_lines + next_lines)
            continue

        # Case 2: a paragraph continues. Last line is normal prose not ending
        # in sentence punctuation, next line starts lowercase prose.
        is_prose_last = (
            last
            and not last.startswith(("#", "-", "|", ">", "*"))
            and not re.search(r"[.!?:;)\]]$", last)
        )
        is_prose_next = bool(first) and bool(re.match(r"^[a-záéíóúñ0-9(]", first))
        if is_prose_last and is_prose_next:
            joined_first = prev_lines[-1].rstrip() + " " + next_lines[0].lstrip()
            merged = "\n".join(prev_lines[:-1] + [joined_first] + next_lines[1:])
            continue

        # Default: normal page break.
        merged = (merged + "\n\n" + nxt).strip()

    return merged


def run_pipeline_b(pdf_path: Path, output_dir: Path, model: str, client, logger) -> dict:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pdf_path.stem}.md"

    result = {
        "pdf_name": pdf_path.name,
        "output_path": str(out_path),
        "total_cost": 0.0,
        "pages_processed": 0,
    }

    # --- Step 1: render pages -------------------------------------------
    page_markdowns: list[str] = []
    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count
        logger.info("Pipeline B: %s has %d page(s)", pdf_path.name, page_count)

        # --- Step 2: per-page extraction --------------------------------
        for i in range(page_count):
            logger.info("Processing page %d/%d", i + 1, page_count)
            try:
                image = _render_page(doc, i, dpi=150)
                b64 = image_to_base64(image)
                text, usage = call_openai_vision(
                    PAGE_SYSTEM_PROMPT, PAGE_USER_PROMPT, b64, model, client
                )
                cost = count_tokens_cost(usage, model)
                result["total_cost"] += cost
                logger.info(
                    "Page %d tokens=%s cost=$%.5f",
                    i + 1, usage.get("total_tokens"), cost,
                )
                page_markdowns.append(_strip_code_fences(text))
                result["pages_processed"] += 1
            except Exception:  # noqa: BLE001
                logger.error(
                    "Page %d extraction failed:\n%s", i + 1, traceback.format_exc()
                )
                page_markdowns.append("")  # keep ordering intact

    # --- Step 3: continuity repair --------------------------------------
    logger.info("Running page-continuity repair")
    stitched = _repair_continuity([p for p in page_markdowns])

    # --- Step 4: cleanup and save ---------------------------------------
    final_md = clean_markdown(stitched)
    out_path.write_text(final_md, encoding="utf-8")
    logger.info(
        "Pipeline B done: %d page(s) -> %s", result["pages_processed"], out_path
    )
    return result
