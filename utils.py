"""Shared helpers for the three PDF -> Markdown extraction pipelines.

Everything path-related is built on top of pathlib.Path. The OpenAI client is
created once by the orchestrator and threaded through to every pipeline, so the
helpers here never instantiate their own client.
"""

from __future__ import annotations

import base64
import io
import logging
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image

# ---------------------------------------------------------------------------
# Pricing (USD per 1,000,000 tokens)
# ---------------------------------------------------------------------------
PRICING = {
    "gpt-4o":      {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


# ---------------------------------------------------------------------------
# PDF / image helpers
# ---------------------------------------------------------------------------
def get_pdf_paths(data_dir: str) -> list[Path]:
    """Recursively scan ``data_dir`` for ``*.pdf`` files, sorted by path."""
    root = Path(data_dir)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


def pdf_page_to_image(pdf_path: Path, page_number: int, dpi: int = 150) -> Image.Image:
    """Render a single (0-indexed) PDF page to a PIL Image at the given DPI."""
    with fitz.open(pdf_path) as doc:
        if page_number < 0 or page_number >= doc.page_count:
            raise IndexError(
                f"page_number {page_number} out of range for {pdf_path} "
                f"({doc.page_count} pages)"
            )
        page = doc.load_page(page_number)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def crop_image_region(image: Image.Image, bbox: tuple) -> Image.Image:
    """Crop a PIL image to ``(x0, y0, x1, y1)`` pixel coordinates.

    The bbox is clamped to the image bounds and normalized so x0<x1, y0<y1.
    """
    x0, y0, x1, y1 = bbox
    x0, x1 = sorted((int(round(x0)), int(round(x1))))
    y0, y1 = sorted((int(round(y0)), int(round(y1))))
    x0 = max(0, min(x0, image.width))
    x1 = max(0, min(x1, image.width))
    y0 = max(0, min(y0, image.height))
    y1 = max(0, min(y1, image.height))
    if x1 <= x0 or y1 <= y0:
        # Degenerate region -> return the whole image rather than crash.
        return image
    return image.crop((x0, y0, x1, y1))


def image_to_base64(image: Image.Image) -> str:
    """Encode a PIL image as a raw (no data-URI prefix) base64 PNG string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# OpenAI vision call
# ---------------------------------------------------------------------------
def call_openai_vision(
    prompt_system: str,
    prompt_user: str,
    image_b64: str,
    model: str,
    client,
) -> tuple[str, dict]:
    """Call Chat Completions with one high-detail base64 PNG image.

    Returns ``(response_text, usage_dict)`` where usage_dict has the keys
    ``prompt_tokens``, ``completion_tokens`` and ``total_tokens``.

    Retries up to 3 times with exponential backoff (2s, 4s, 8s) on any error.
    """
    messages = [
        {"role": "system", "content": prompt_system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_user},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]

    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            text = response.choices[0].message.content or ""
            usage = response.usage
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }
            return text.strip(), usage_dict
        except Exception as exc:  # noqa: BLE001 - surfaced after retries
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))  # 2s, 4s, 8s

    raise RuntimeError(f"OpenAI vision call failed after 3 attempts: {last_error}")


def count_tokens_cost(usage_dict: dict, model: str) -> float:
    """Compute USD cost for a usage dict given the model's pricing."""
    pricing = PRICING.get(model)
    if pricing is None:
        # Tolerate dated model ids like "gpt-4o-2024-08-06".
        for known, value in PRICING.items():
            if model.startswith(known):
                pricing = value
                break
    if pricing is None:
        return 0.0

    prompt_tokens = usage_dict.get("prompt_tokens", 0) or 0
    completion_tokens = usage_dict.get("completion_tokens", 0) or 0
    cost = (
        prompt_tokens / 1_000_000 * pricing["input"]
        + completion_tokens / 1_000_000 * pricing["output"]
    )
    return cost


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(pipeline_name: str, pdf_name: str, log_dir: Path) -> logging.Logger:
    """Logger writing INFO+ to console and DEBUG+ to logs/{pipeline}_{pdf}.log."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    safe_pdf = Path(pdf_name).stem
    logger_name = f"{pipeline_name}.{safe_pdf}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Reset handlers so re-runs in the same process don't duplicate output.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_path = log_dir / f"{pipeline_name}_{safe_pdf}.log"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Markdown cleanup
# ---------------------------------------------------------------------------
def clean_markdown(text: str) -> str:
    """Strip repeated headers/footers, collapse blank lines, trim line ends.

    - Lines that appear 3+ times *identically* across the document are treated
      as recurring page headers/footers and removed. Table rows/separators
      (lines starting with ``|``) are never removed by this rule.
    - Runs of 3+ blank lines collapse to a maximum of two.
    - Trailing whitespace is stripped from every line.
    """
    if not text:
        return ""

    raw_lines = text.splitlines()

    # Count identical non-empty, non-table lines.
    counts = Counter()
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("|"):
            continue
        counts[stripped] += 1

    repeated = {line for line, n in counts.items() if n >= 3}

    kept: list[str] = []
    for line in raw_lines:
        if line.strip() in repeated:
            continue
        kept.append(line.rstrip())

    # Collapse 3+ consecutive blank lines down to two.
    out: list[str] = []
    blank_run = 0
    for line in kept:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                out.append(line)
        else:
            blank_run = 0
            out.append(line)

    return "\n".join(out).strip() + "\n"


# ---------------------------------------------------------------------------
# Small markdown-table utilities shared by pipelines A and C
# ---------------------------------------------------------------------------
def split_table_row(row: str) -> list[str]:
    """Split a markdown table row into trimmed cell strings."""
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def is_separator_row(row: str) -> bool:
    """True if a row looks like a GFM header separator (``|---|---|``)."""
    cells = split_table_row(row)
    if not cells:
        return False
    return all(set(c) <= set("-: ") and "-" in c for c in cells if c != "")
