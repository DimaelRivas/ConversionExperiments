"""Generates human-readable review checklists from flagged tables.

Two outputs:
  - one {stem}_checklist.md per PDF with per-table instructions
  - one _master_checklist.md summarizing every PDF in a single place
A reviewer should be able to work entirely from these files.

Flagged entries are plain dicts (from pipeline.py) with keys:
    start_line, end_line, reason, heuristic, col_count,
    page_number (0-indexed, may be None), table_index_on_page (may be None),
    section_title (str)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

ISSUE_LINES = {
    "image_proximity": "Image elements detected immediately before this table (image_proximity)",
    "split_table": "Table appears split — consecutive table with same column count detected (split_table)",
}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _page_human(entry: dict) -> str:
    pn = entry.get("page_number")
    return str(pn + 1) if pn is not None else "?"


def generate_pdf_checklist(
    pdf_name: str,
    flagged_tables: list[dict],
    output_path: Path,
    total_tables: int | None = None,
    total_pages: int | None = None,
) -> Path:
    """Write logs/checklists/{stem}_checklist.md and return its path."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_name).stem
    out_file = output_path / f"{stem}_checklist.md"

    # Distinct tables flagged (a table can be flagged by both heuristics)
    distinct = {e["start_line"] for e in flagged_tables}
    flagged_count = len(distinct)
    clean_count = max(0, total_tables - flagged_count) if total_tables is not None else 0

    lines: list[str] = [
        f"# Review Checklist — {pdf_name}",
        "",
        f"Generated: {_now()}",
        f"Total tables flagged: {flagged_count}",
        f"Total tables clean: {clean_count}",
        "",
        "---",
        "",
        "## Tables Requiring Review",
        "",
    ]

    if not flagged_tables:
        lines.append("_No tables were flagged — this document needs no table review._")
        lines.append("")
    else:
        ordered = sorted(flagged_tables, key=lambda e: (e["start_line"], e.get("heuristic", "")))
        for entry in ordered:
            page_human = _page_human(entry)
            table_human = (
                (entry["table_index_on_page"] + 1)
                if entry.get("table_index_on_page") is not None else "?"
            )
            section = entry.get("section_title") or ""
            title_suffix = f" ({section})" if section else ""
            heuristic = entry.get("heuristic", "")
            issue = ISSUE_LINES.get(heuristic, entry.get("reason", ""))

            lines.append(f"### 🔴 Page {page_human} — Table {table_human}{title_suffix}")
            lines.append("")
            lines.append(f"**Issue:** {issue}")
            if heuristic == "split_table":
                lines.append(
                    f"**Action:** Open {pdf_name} at page {page_human}. Merge this "
                    f"fragment with the adjacent {entry.get('col_count', '?')}-column table "
                    f"into a single table."
                )
            else:
                lines.append(
                    f"**Action:** Open {pdf_name} at page {page_human}. Reconstruct the "
                    f"table, restoring any header rows that were extracted as images."
                )
                lines.append(
                    "**Note:** This table's header row(s) likely contained images. "
                    "Replace any image cells with a text description or placeholder."
                )
            lines.append("")
            lines.append("In the output file, search for:")
            lines.append("```")
            lines.append(f"<!-- ⚠️ REVIEW NEEDED ... Page: {page_human} -->")
            lines.append("```")
            lines.append("")
            lines.append("---")
            lines.append("")

    lines.append("## Clean Tables (No Review Needed)")
    lines.append("")
    if total_pages:
        flagged_pages = {
            e["page_number"] + 1 for e in flagged_tables if e.get("page_number") is not None
        }
        clean_pages = [p for p in range(1, total_pages + 1) if p not in flagged_pages]
        clean_str = ", ".join(str(p) for p in clean_pages) if clean_pages else "(none)"
        lines.append(f"Pages with no flagged tables: {clean_str}")
    else:
        lines.append(f"Clean tables: {clean_count}")
    lines.append("")

    out_file.write_text("\n".join(lines), encoding="utf-8")
    return out_file


def generate_master_checklist(all_results: list[dict], output_path: Path) -> Path:
    """Write logs/checklists/_master_checklist.md summarizing all PDFs."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / "_master_checklist.md"

    lines: list[str] = [
        "# Master Review Checklist",
        "",
        f"Generated: {_now()}",
        "",
        "| PDF | Total Tables | Flagged | Clean | Checklist |",
        "|---|---|---|---|---|",
    ]

    tot_tables = tot_flagged = tot_clean = 0
    for r in all_results:
        if r.get("skipped"):
            continue
        name = r.get("pdf_name", "?")
        stem = Path(name).stem
        t = r.get("tables_found", 0)
        f = r.get("tables_flagged", 0)
        c = r.get("tables_clean", 0)
        tot_tables += t
        tot_flagged += f
        tot_clean += c
        lines.append(f"| {name} | {t} | {f} | {c} | [link]({stem}_checklist.md) |")

    lines.append(f"| **TOTAL** | **{tot_tables}** | **{tot_flagged}** | **{tot_clean}** | |")
    lines.append("")
    lines.append("## All Flagged Tables")
    lines.append("")
    lines.append("| PDF | Page | Table | Heuristic | Issue |")
    lines.append("|---|---|---|---|---|")

    any_flagged = False
    for r in all_results:
        if r.get("skipped"):
            continue
        name = r.get("pdf_name", "?")
        for d in r.get("flagged_details", []):
            any_flagged = True
            page = d.get("page_number")
            page_str = str(page + 1) if page is not None else "?"
            tidx = d.get("table_index_on_page")
            tidx_str = str(tidx + 1) if tidx is not None else "?"
            lines.append(
                f"| {name} | {page_str} | {tidx_str} | {d.get('heuristic', '')} | {d.get('reason', '')} |"
            )

    if not any_flagged:
        lines.append("| — | — | — | — | No tables flagged across any PDF |")

    lines.append("")
    out_file.write_text("\n".join(lines), encoding="utf-8")
    return out_file
