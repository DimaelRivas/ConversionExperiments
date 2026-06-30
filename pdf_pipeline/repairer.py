"""LLM-based table repair using tightly cropped page region images.

The LLM is NEVER called with a full page — only with a cropped rectangle
containing exactly one table (plus 10px padding on each side).
"""
from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)

# Exact system prompt from spec — do not paraphrase.
_SYSTEM_PROMPT = """You are a precise document extraction assistant specializing in table reconstruction.
Your task is to extract a single table from the provided image and reproduce it as a valid GitHub-Flavored Markdown (GFM) table.

Rules:
- Reproduce EVERY row and EVERY column exactly as they appear visually.
- For tables with multi-level headers (a header row that spans multiple sub-columns), represent the top-level header by repeating its label in each sub-column it covers. Example: if "RECETAS MAT" spans 3 columns, the header row should have "RECETAS MAT" in each of those 3 column positions.
- Do not skip rows, merge rows, or summarize content.
- Numeric values must be exact: preserve currency symbols, percentage signs, decimal separators, and thousand separators exactly as shown.
- Empty cells must be represented as a single space: | |
- If a cell contains a list, use semicolons to separate items within the cell.
- Output ONLY the markdown table. No preamble, no explanation, no code fences, no commentary.
- The document language is Spanish. Preserve all text exactly — do not translate."""

PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":      {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output":  0.60},
}


def render_page_region(
    pdf_path: Path,
    page_number: int,
    bbox: tuple | None,
    dpi: int = 200,
) -> Image.Image:
    """Crop-render a single table region from a PDF page.

    Parameters
    ----------
    bbox : (x0, y0, x1, y1) in PDF points, PyMuPDF top-left origin.
           Pass None to render the full page at 150 DPI (rare fallback).
    dpi  : rendering resolution. 200 DPI gives crisp table borders and text.
    """
    with fitz.open(str(pdf_path)) as doc:
        page = doc.load_page(page_number)

        if bbox is None:
            zoom = 150 / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        zoom = dpi / 72.0
        # 10-pixel padding expressed in PDF points at this zoom level
        pad_pt = 10.0 / zoom
        x0, y0, x1, y1 = bbox
        x0 = max(0.0, x0 - pad_pt)
        y0 = max(0.0, y0 - pad_pt)
        x1 = min(float(page.rect.width),  x1 + pad_pt)
        y1 = min(float(page.rect.height), y1 + pad_pt)

        clip = fitz.Rect(x0, y0, x1, y1)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def image_to_base64(image: Image.Image) -> str:
    """Encode a PIL Image as a raw base64 PNG string (no data-URI prefix)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def repair_table(
    pdf_path: Path,
    page_number: int,
    bbox: tuple | None,
    model: str,
    client,
    context: str = "",
) -> tuple[str, dict]:
    """Send a cropped table image to the vision LLM and return GFM markdown.

    Returns (repaired_markdown, usage_dict).
    On complete failure returns ("", {}) after 3 retries with exponential backoff.
    """
    image = render_page_region(pdf_path, page_number, bbox)
    b64 = image_to_base64(image)

    if context.strip():
        context_block = (
            "\nFor reference, the surrounding document context is:\n"
            f"<context>\n{context}\n</context>"
        )
    else:
        context_block = ""

    user_prompt = (
        f"Extract the table in this image as a GitHub-Flavored Markdown table.{context_block}"
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            text = (response.choices[0].message.content or "").strip()
            text = _strip_fences(text)
            u = response.usage
            usage_dict = {
                "prompt_tokens":     getattr(u, "prompt_tokens",     0) or 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                "total_tokens":      getattr(u, "total_tokens",      0) or 0,
            }
            return text, usage_dict
        except Exception as exc:
            last_exc = exc
            logger.warning("Repair attempt %d/3 failed: %s", attempt + 1, exc)
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))  # 2s, 4s

    logger.error(
        "All repair attempts failed for page %d bbox %s: %s",
        page_number, bbox, last_exc,
    )
    return "", {}


def calculate_cost(usage_dict: dict, model: str) -> float:
    """Compute USD cost from a usage dict. Returns 0.0 for unknown models."""
    if not usage_dict:
        return 0.0
    pricing = PRICING.get(model)
    if pricing is None:
        # Accept version-stamped ids like "gpt-4o-mini-2024-07-18"
        for known, value in PRICING.items():
            if model.startswith(known):
                pricing = value
                break
    if pricing is None:
        return 0.0
    p = usage_dict.get("prompt_tokens", 0) or 0
    c = usage_dict.get("completion_tokens", 0) or 0
    return p / 1_000_000 * pricing["input"] + c / 1_000_000 * pricing["output"]


def _strip_fences(text: str) -> str:
    """Remove a leading/trailing ```...``` block if the model added one."""
    lines = text.strip().splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
