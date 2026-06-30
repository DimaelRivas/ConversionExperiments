"""CLI entry point for the marker-based PDF → Markdown review pipeline.

Usage:
    python run.py [--pdf filename.pdf] [--force] [--test]
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR      = ROOT / "data"
OUTPUT_DIR    = ROOT / "output"
LOG_DIR       = ROOT / "logs"
CHECKLIST_DIR = LOG_DIR / "checklists"


def _ensure_dirs() -> None:
    for d in (OUTPUT_DIR, LOG_DIR, CHECKLIST_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("pipeline_run")
    logger.setLevel(logging.DEBUG)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(LOG_DIR / "pipeline_run.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marker-based PDF → Markdown review pipeline")
    p.add_argument("--pdf",   default=None, help="Process only this PDF filename (must be in data/)")
    p.add_argument("--force", action="store_true", help="Reprocess even if output exists")
    p.add_argument("--test",  action="store_true", help="Run self-verification after the pipeline")
    return p.parse_args()


def _get_pdfs(pdf_arg: str | None) -> list[Path]:
    all_pdfs = sorted(DATA_DIR.rglob("*.pdf"))
    if pdf_arg is None:
        return all_pdfs
    wanted = Path(pdf_arg).name
    return [p for p in all_pdfs if p.name == wanted]


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

def _render_summary(results: list[dict]) -> str:
    headers = ("PDF", "Tables", "Flagged", "Images", "Splits", "Clean")
    rows: list[tuple[str, ...]] = []
    tot_tables = tot_flagged = tot_img = tot_split = tot_clean = 0

    for r in results:
        if r.get("skipped"):
            continue
        t = r.get("tables_found", 0)
        f = r.get("tables_flagged", 0)
        img = r.get("tables_flagged_image", 0)
        spl = r.get("tables_flagged_split", 0)
        clean = r.get("tables_clean", 0)
        tot_tables += t
        tot_flagged += f
        tot_img += img
        tot_split += spl
        tot_clean += clean
        rows.append((r.get("pdf_name", "?"), str(t), str(f), str(img), str(spl), str(clean)))

    total_row = (
        "TOTAL", str(tot_tables), str(tot_flagged),
        str(tot_img), str(tot_split), str(tot_clean),
    )
    all_rows = [headers] + rows + [total_row]
    widths = [max(len(r[i]) for r in all_rows) + 2 for i in range(len(headers))]

    def _fmt(cells, mid="║"):
        return "║" + mid.join(str(c).center(widths[i]) for i, c in enumerate(cells)) + "║"

    def _rule(left, m, right, ch="═"):
        return left + m.join(ch * w for w in widths) + right

    total_width = sum(widths) + len(widths) + 1
    lines = [
        "╔" + "═" * (total_width - 2) + "╗",
        "║" + " PIPELINE SUMMARY ".center(total_width - 2) + "║",
        _rule("╠", "╦", "╣"),
        _fmt(headers),
        _rule("╠", "╬", "╣"),
    ]
    lines += [_fmt(r) for r in rows]
    lines += [_rule("╠", "╬", "╣"), _fmt(total_row), _rule("╚", "╩", "╝")]
    return "\n".join(lines)


def _render_flagged_list(results: list[dict]) -> list[str]:
    out: list[str] = []
    for r in results:
        for d in r.get("flagged_details", []):
            heuristic = d.get("heuristic", "")
            section = d.get("section_title") or ""
            page = d.get("page_number")
            locator = f'near "{section}"' if section else (
                f"page {page + 1}" if page is not None else "(location unknown)"
            )
            out.append(f"  → {r['pdf_name']}  [{heuristic}]  {locator}")
            if heuristic == "image_proximity":
                hint = "Header row(s) likely contained images"
            elif heuristic == "split_table":
                hint = f"Two fragments with {d.get('col_count', '?')} columns, merge them"
            else:
                hint = d.get("reason", "")
            out.append(f"                                  {hint}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    if not DATA_DIR.exists():
        print(f"ERROR: data directory not found: {DATA_DIR}", file=sys.stderr)
        return 1

    pdfs = _get_pdfs(args.pdf)
    if not pdfs:
        print(
            f"No PDFs found in {DATA_DIR}"
            + (f" matching '{args.pdf}'" if args.pdf else "")
            + ".\nPlace PDF files in data/ and retry.",
            file=sys.stderr,
        )
        return 1

    _ensure_dirs()
    logger = _setup_logger()

    from pipeline import run_pipeline
    from checklist import generate_master_checklist

    all_results: list[dict] = []
    all_success = True

    for pdf_path in pdfs:
        out_md = OUTPUT_DIR / f"{pdf_path.stem}.md"
        if out_md.exists() and not args.force:
            logger.info("SKIP (exists — use --force): %s", pdf_path.name)
            all_results.append({"pdf_name": pdf_path.name, "skipped": True, "flagged_details": []})
            continue

        logger.info("Processing %s", pdf_path.name)
        try:
            res = run_pipeline(pdf_path, OUTPUT_DIR, CHECKLIST_DIR, logger)
            all_results.append(res)
        except Exception:
            all_success = False
            tb = traceback.format_exc()
            logger.error("Pipeline FAILED for %s:\n%s", pdf_path.name, tb)
            all_results.append(
                {
                    "pdf_name": pdf_path.name,
                    "tables_found": 0,
                    "tables_flagged": 0,
                    "tables_clean": 0,
                    "flagged_details": [],
                    "error": tb,
                }
            )

    # Master checklist across all PDFs
    generate_master_checklist(all_results, CHECKLIST_DIR)

    # Final report
    processed = [r for r in all_results if not r.get("skipped")]
    print()
    print(_render_summary(all_results))
    flagged_lines = _render_flagged_list(processed)
    if flagged_lines:
        print("\nFlagged tables requiring human review:")
        print("\n".join(flagged_lines))
    else:
        print("\nNo tables flagged — nothing requires human review.")
    print(f"\nOutput markdown files:  {OUTPUT_DIR}/")
    print(f"Review checklists:      {CHECKLIST_DIR}/")
    print(f"Full run log:           {LOG_DIR / 'pipeline_run.log'}")

    # Optional self-verification
    if args.test:
        print("\nRunning self-verification test...")
        rc = subprocess.call([sys.executable, str(ROOT / "test_validation_case.py")], cwd=str(ROOT))
        if rc != 0:
            all_success = False

    return 0 if all_success else 2


if __name__ == "__main__":
    sys.exit(main())
