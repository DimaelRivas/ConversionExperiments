"""Markdown-only table problem scanner (replaces the PyMuPDF validator).

PyMuPDF's find_tables() reads colored background rectangles in marketing PDFs as
table cells, producing nonsensical geometry and false positives on every correct
table. This scanner instead detects the two failure patterns Docling actually
exhibits, directly from its markdown output — no PDF access, no external libs.

Pattern 1 (image_proximity): a table preceded by floating <!-- image --> elements
    whose header rows contained images Docling could not turn into text.
Pattern 2 (split_table): one source table emitted as two consecutive table blocks
    with the same column count and nothing but blanks/images between them.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

IMAGE_TOKEN = "<!-- image -->"
_SEP_RE = re.compile(r"^\|[\s\-:|]+\|?$")  # |---|---| style separator row


@dataclass
class FlaggedTable:
    start_line: int       # 0-indexed line where the table starts (first | line)
    end_line: int         # 0-indexed line of the last | line in the table
    col_count: int        # number of columns in this table
    heuristic: str        # "image_proximity" or "split_table"
    reason: str           # human-readable explanation for the checklist
    partner_start: int    # split_table: start_line of the other fragment; else -1


def _is_table_line(line: str) -> bool:
    return line.strip().startswith("|")


def _is_separator_line(line: str) -> bool:
    s = line.strip()
    if not s.startswith("|"):
        return False
    if _SEP_RE.match(s):
        # Must actually contain a dash (avoid matching e.g. "| |")
        return "-" in s
    return False


def _count_columns(cells_line: str) -> int:
    s = cells_line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return len([c for c in s.split("|")])


def parse_tables(lines: list[str]) -> list[dict]:
    """First pass: every table in the document.

    A table is a maximal run of consecutive lines that each start with `|`.
    Returns dicts with start_line, end_line, col_count — sorted by start_line.
    """
    tables: list[dict] = []
    i, n = 0, len(lines)
    while i < n:
        if _is_table_line(lines[i]):
            start = i
            while i < n and _is_table_line(lines[i]):
                i += 1
            end = i - 1

            # Column count: prefer the separator row, else the first row.
            sep_idx = None
            for j in range(start, end + 1):
                if _is_separator_line(lines[j]):
                    sep_idx = j
                    break
            count_line = lines[sep_idx] if sep_idx is not None else lines[start]
            col_count = _count_columns(count_line)

            tables.append(
                {"start_line": start, "end_line": end, "col_count": col_count}
            )
        else:
            i += 1

    tables.sort(key=lambda t: t["start_line"])
    return tables


def scan_for_image_proximity(lines: list[str], tables: list[dict]) -> list[FlaggedTable]:
    """Heuristic 1: flag a table if an <!-- image --> sits in the 6 lines above it."""
    flagged: list[FlaggedTable] = []
    for t in tables:
        start = t["start_line"]
        window = lines[max(0, start - 6) : start]
        if any(ln.strip() == IMAGE_TOKEN for ln in window):
            flagged.append(
                FlaggedTable(
                    start_line=start,
                    end_line=t["end_line"],
                    col_count=t["col_count"],
                    heuristic="image_proximity",
                    reason=(
                        "One or more image elements appear immediately before this "
                        "table. The table's header row(s) likely contained images that "
                        "docling could not extract as text. Check the source PDF for "
                        "missing rows."
                    ),
                    partner_start=-1,
                )
            )
    return flagged


def scan_for_split_tables(lines: list[str], tables: list[dict]) -> list[FlaggedTable]:
    """Heuristic 2: flag both fragments of a table split into adjacent blocks."""
    flagged: dict[int, FlaggedTable] = {}  # start_line -> flag, prevents double-flagging

    for a, b in zip(tables, tables[1:]):
        between = lines[a["end_line"] + 1 : b["start_line"]]
        only_blank_or_image = all(
            ln.strip() == "" or ln.strip() == IMAGE_TOKEN for ln in between
        )
        if not only_blank_or_image:
            continue
        if a["col_count"] != b["col_count"]:
            continue

        n_cols = a["col_count"]
        reason = (
            f"This table appears to be a fragment of a larger table split by docling. "
            f"A consecutive table with the same column count ({n_cols} columns) follows "
            f"with no content between them. Merge both fragments into one table in the "
            f"source PDF."
        )

        # Flag A (partner = B) and B (partner = A), first pairing wins.
        if a["start_line"] not in flagged:
            flagged[a["start_line"]] = FlaggedTable(
                start_line=a["start_line"],
                end_line=a["end_line"],
                col_count=n_cols,
                heuristic="split_table",
                reason=reason,
                partner_start=b["start_line"],
            )
        if b["start_line"] not in flagged:
            flagged[b["start_line"]] = FlaggedTable(
                start_line=b["start_line"],
                end_line=b["end_line"],
                col_count=n_cols,
                heuristic="split_table",
                reason=reason,
                partner_start=a["start_line"],
            )

    return list(flagged.values())


def scan_markdown(markdown: str) -> list[FlaggedTable]:
    """Main entry point. Returns all flagged tables, sorted by start_line."""
    lines = markdown.splitlines()
    tables = parse_tables(lines)

    flagged: list[FlaggedTable] = []
    flagged += scan_for_image_proximity(lines, tables)
    flagged += scan_for_split_tables(lines, tables)

    # A table flagged by both heuristics keeps both flags (different problems).
    flagged.sort(key=lambda f: f.start_line)

    n_image = sum(1 for f in flagged if f.heuristic == "image_proximity")
    n_split = sum(1 for f in flagged if f.heuristic == "split_table")
    logger.info(
        "Scanner: %d tables found, %d flagged (%d image_proximity, %d split_table)",
        len(tables), len(flagged), n_image, n_split,
    )

    return flagged
