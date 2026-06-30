"""Writes human-readable diff reports for every repaired table.

These are the primary review artifacts for the human-in-the-loop phase.
"""
from __future__ import annotations

import difflib
from pathlib import Path


def log_table_repair(
    pdf_name: str,
    page_number: int,       # 0-indexed
    table_index: int,       # 0-indexed, position on the page
    reason: str,
    original_markdown: str,
    repaired_markdown: str,
    log_dir: Path,
    usage_dict: dict | None = None,
    cost: float = 0.0,
    docling_cols: int = 0,
    docling_rows: int = 0,
    pymupdf_cols: int = 0,
    pymupdf_rows: int = 0,
) -> Path:
    """Write logs/repairs/{pdf_stem}_page{N}_table{M}.md and return the path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    pdf_stem = Path(pdf_name).stem
    filename = f"{pdf_stem}_page{page_number + 1}_table{table_index + 1}.md"
    out_path = log_dir / filename

    usage_dict = usage_dict or {}
    prompt_tokens     = usage_dict.get("prompt_tokens",     0) or 0
    completion_tokens = usage_dict.get("completion_tokens", 0) or 0
    total_tokens      = usage_dict.get("total_tokens",      0) or 0

    orig_lines = original_markdown.strip().splitlines() if original_markdown.strip() else []
    rep_lines  = repaired_markdown.strip().splitlines() if repaired_markdown.strip() else []

    # Build a clean diff block using unified_diff
    raw_diff = list(difflib.unified_diff(orig_lines, rep_lines, lineterm="", n=0))
    diff_lines: list[str] = []
    for dl in raw_diff:
        if dl.startswith("---") or dl.startswith("+++"):
            continue
        if dl.startswith("@@"):
            diff_lines.append("  ...")
        elif dl.startswith("-"):
            diff_lines.append(dl)
        elif dl.startswith("+"):
            diff_lines.append(dl)
        else:
            diff_lines.append(" " + dl)

    diff_block = "\n".join(diff_lines) if diff_lines else "  (no diff — original and repaired are identical)"

    orig_section = original_markdown.strip() if original_markdown.strip() else "_empty (orphan table)_"
    rep_section  = repaired_markdown.strip() if repaired_markdown.strip() else "_Repair failed — LLM returned empty response_"

    content = f"""# Table Repair Report

**PDF:** {pdf_name}
**Page:** {page_number + 1} (0-indexed: {page_number})
**Table index on page:** {table_index + 1}
**Repair reason:** {reason}

---

## Validation Failure

| Metric | Docling | PyMuPDF |
|---|---|---|
| Columns | {docling_cols} | {pymupdf_cols} |
| Rows | {docling_rows} | {pymupdf_rows} |

---

## Original (Docling Output)

{orig_section}

---

## Repaired (LLM Output)

{rep_section}

---

## Line-by-Line Diff

```diff
{diff_block}
```

---

## Token Usage

- Prompt tokens: {prompt_tokens:,}
- Completion tokens: {completion_tokens:,}
- Total tokens: {total_tokens:,}
- Estimated cost: ${cost:.5f}
"""

    out_path.write_text(content, encoding="utf-8")
    return out_path


def write_repair_summary(repairs: list[dict], log_dir: Path) -> Path:
    """Write logs/repairs/_summary.md — all repairs in a single table."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "_summary.md"

    rows: list[str] = []
    for r in repairs:
        reason_short = (r.get("reason") or "").split(";")[0][:60]
        rows.append(
            f"| {r.get('pdf_name', '')} "
            f"| {r.get('page_number', 0) + 1} "
            f"| {r.get('table_index', 0) + 1} "
            f"| {reason_short} "
            f"| {r.get('cols_before', 0)} "
            f"| {r.get('cols_after', 0)} "
            f"| ${r.get('cost', 0.0):.5f} |"
        )

    table_md = (
        "| PDF | Page | Table | Reason | Columns Before | Columns After | Cost |\n"
        "|---|---|---|---|---|---|---|\n"
        + ("\n".join(rows) if rows else "| — | — | — | No repairs made | — | — | — |")
    )

    content = f"# Repair Summary\n\n{table_md}\n"
    summary_path.write_text(content, encoding="utf-8")
    return summary_path
