"""Structural cross-validation of docling tables using PyMuPDF find_tables().

Responsibility: compare *structure only* — column counts, row counts, bounding boxes.
Never accesses table cell content. Never modifies anything.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class TableStructure:
    page_number: int   # 0-indexed
    table_index: int   # 0-indexed, position among tables on this page
    col_count: int
    row_count: int
    bbox: tuple        # (x0, y0, x1, y1) PDF points, PyMuPDF top-left origin


def get_page_table_structures(pdf_path: Path, page_number: int) -> list[TableStructure]:
    """Return PyMuPDF geometric table info for a single page (0-indexed).

    Returns an empty list (with a warning) if find_tables() is unavailable or
    raises; callers must handle this gracefully.
    """
    try:
        with fitz.open(str(pdf_path)) as doc:
            if page_number < 0 or page_number >= doc.page_count:
                return []
            page = doc.load_page(page_number)
            result = page.find_tables()
            structures = []
            for i, table in enumerate(result.tables):
                structures.append(
                    TableStructure(
                        page_number=page_number,
                        table_index=i,
                        col_count=int(table.col_count),
                        row_count=int(table.row_count),
                        bbox=tuple(float(v) for v in table.bbox),
                    )
                )
            return structures
    except Exception as exc:
        logger.warning(
            "find_tables() failed on page %d of %s: %s", page_number, Path(pdf_path).name, exc
        )
        return []


def parse_docling_table_structure(markdown_table: str) -> tuple[int, int]:
    """Parse a GFM table string and return (col_count, data_row_count).

    col_count   – columns in the first non-separator row
    row_count   – data rows only (excludes the header row and the |---| separator)
    """
    lines = [ln for ln in markdown_table.strip().splitlines() if ln.strip()]
    if not lines:
        return 0, 0

    # Locate the separator row (|---|---|)
    sep_idx: int | None = None
    for i, line in enumerate(lines):
        cells = _split_row(line)
        if cells and all(_is_sep_cell(c) for c in cells if c):
            sep_idx = i
            break

    col_count = len(_split_row(lines[0])) if lines else 0

    if sep_idx is not None:
        row_count = max(0, len(lines) - sep_idx - 1)
    else:
        row_count = max(0, len(lines) - 1)

    return col_count, row_count


def validate_table(
    docling_col_count: int,
    docling_row_count: int,
    pymupdf_structure: TableStructure,
) -> tuple[bool, str]:
    """Return (is_valid, reason).

    Valid when:
    - abs(docling_cols - pymupdf_cols) <= 1   (off-by-one tolerance for header merging)
    - docling_rows >= pymupdf_rows * 0.75     (25% tolerance for multi-row headers)
    """
    col_ok = abs(docling_col_count - pymupdf_structure.col_count) <= 1
    row_ok = (
        pymupdf_structure.row_count == 0
        or docling_row_count >= pymupdf_structure.row_count * 0.75
    )

    if col_ok and row_ok:
        logger.debug(
            "OK: doc(%d cols, %d rows) ≈ pymupdf(%d cols, %d rows)",
            docling_col_count, docling_row_count,
            pymupdf_structure.col_count, pymupdf_structure.row_count,
        )
        return True, ""

    parts: list[str] = []
    if not col_ok:
        parts.append(
            f"Column mismatch: docling={docling_col_count}, "
            f"pymupdf={pymupdf_structure.col_count} "
            f"on page {pymupdf_structure.page_number + 1}, "
            f"table {pymupdf_structure.table_index + 1}"
        )
    if not row_ok:
        parts.append(
            f"Row mismatch: docling={docling_row_count}, "
            f"pymupdf={pymupdf_structure.row_count} "
            f"on page {pymupdf_structure.page_number + 1}, "
            f"table {pymupdf_structure.table_index + 1}"
        )

    reason = "; ".join(parts)
    logger.debug("FAIL: %s", reason)
    return False, reason


def find_orphan_tables(
    pdf_path: Path,
    page_number: int,
    docling_table_count: int,
) -> int:
    """Count tables PyMuPDF found on the page that docling did not extract."""
    structures = get_page_table_structures(pdf_path, page_number)
    return max(0, len(structures) - docling_table_count)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_row(row: str) -> list[str]:
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_sep_cell(cell: str) -> bool:
    """True if the cell content is a GFM separator (only -, : and spaces)."""
    s = cell.strip()
    return bool(s) and set(s) <= set("-: ")
