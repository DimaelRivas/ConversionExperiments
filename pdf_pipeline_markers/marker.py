"""Inserts HTML-comment review markers into the markdown at flagged tables.

Markers are HTML comments, so they are invisible in rendered markdown but easy
for a reviewer to search for. They wrap the suspect table without altering it.

Marker entries are plain dicts produced by pipeline.py with the keys:
    start_line, end_line, reason
    page_number (0-indexed, from docling JSON; may be None)
    table_index_on_page (0-indexed; may be None)
"""
from __future__ import annotations

MARKER_TEMPLATE = """\
<!-- ⚠️ REVIEW NEEDED
     PDF: {pdf_name}
     Page: {page_human} (0-indexed: {page_zero})
     Table: {table_index} on this page
     Issue: {reason}
     Action: Open source PDF at page {page_human} and manually correct the table below.
-->"""

CLOSING_MARKER = "<!-- END REVIEW SECTION -->"


def _format_marker(entry: dict, pdf_name: str) -> str:
    page_number = entry.get("page_number")
    table_index = entry.get("table_index_on_page")
    page_human = (page_number + 1) if page_number is not None else "?"
    page_zero = page_number if page_number is not None else "?"
    table_human = (table_index + 1) if table_index is not None else "?"
    return MARKER_TEMPLATE.format(
        pdf_name=pdf_name,
        page_human=page_human,
        page_zero=page_zero,
        table_index=table_human,
        reason=entry.get("reason", ""),
    )


def insert_marker(
    document_lines: list[str],
    table_start_line: int,
    entry: dict,
    pdf_name: str,
    table_end_line: int | None = None,
) -> list[str]:
    """Insert an opening marker before the table and a closing marker after it.

    Returns a new list; the input list is not mutated.
    """
    lines = list(document_lines)
    if table_end_line is not None:
        lines.insert(table_end_line + 1, CLOSING_MARKER)
    lines.insert(table_start_line, _format_marker(entry, pdf_name))
    return lines


def apply_all_markers(
    document: str,
    flagged_tables: list[dict],
    pdf_name: str,
) -> str:
    """Apply markers for every flagged table, bottom-to-top to keep indices valid.

    Each entry needs: start_line, end_line, reason (+ optional page_number,
    table_index_on_page for the Page line).
    """
    lines = document.splitlines()

    for entry in sorted(flagged_tables, key=lambda e: e["start_line"], reverse=True):
        start = entry["start_line"]
        end = entry["end_line"]
        # Closing marker after the table (insert higher index first so the
        # opening insert does not shift it).
        lines.insert(end + 1, CLOSING_MARKER)
        lines.insert(start, _format_marker(entry, pdf_name))

    return "\n".join(lines)
