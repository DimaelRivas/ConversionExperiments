"""CLI entry point for the PDF-to-Markdown pipeline.

Usage:
    python run.py [--pdf filename.pdf] [--force] [--test]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DATA_DIR    = ROOT / "data"
OUTPUT_DIR  = ROOT / "output"
LOG_DIR     = ROOT / "logs"
REPAIRS_DIR = LOG_DIR / "repairs"
TMP_DIR     = ROOT / "tmp"


def _ensure_dirs() -> None:
    for d in (OUTPUT_DIR, LOG_DIR, REPAIRS_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _setup_root_logger() -> logging.Logger:
    log_path = LOG_DIR / "pipeline_run.log"
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
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PDF → Markdown production pipeline")
    p.add_argument("--pdf",   default=None, help="Process only this PDF filename (must be in data/)")
    p.add_argument("--force", action="store_true", help="Overwrite existing output")
    p.add_argument("--test",  action="store_true", help="Run self-verification test after pipeline")
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
    """Render the box-drawing summary table to a string."""
    if not results:
        return "(no PDFs processed)"

    # Collect rows
    rows: list[tuple[str, ...]] = []
    totals = {"pages": 0, "tables": 0, "repaired": 0, "orphans": 0, "cost": 0.0}

    for r in results:
        rows.append((
            r.get("pdf_name", "?"),
            str(r.get("pages_processed", 0)),
            str(r.get("tables_found", 0)),
            str(r.get("tables_repaired", 0)),
            str(r.get("tables_orphaned", 0)),
            f"${r.get('total_cost', 0.0):.3f}",
        ))
        totals["pages"]    += r.get("pages_processed", 0)
        totals["tables"]   += r.get("tables_found", 0)
        totals["repaired"] += r.get("tables_repaired", 0)
        totals["orphans"]  += r.get("tables_orphaned", 0)
        totals["cost"]     += r.get("total_cost", 0.0)

    total_row = (
        "TOTAL",
        str(totals["pages"]),
        str(totals["tables"]),
        str(totals["repaired"]),
        str(totals["orphans"]),
        f"${totals['cost']:.3f}",
    )

    headers = ("PDF", "Pages", "Tables", "Repaired", "Orphans", "Cost")
    all_rows = [headers] + rows + [total_row]
    widths = [max(len(r[i]) for r in all_rows) + 2 for i in range(len(headers))]

    def _fmt(cells: tuple[str, ...], mid: str = "║") -> str:
        return "║" + mid.join(c.center(widths[i]) for i, c in enumerate(cells)) + "║"

    def _rule(left: str, m: str, right: str, ch: str = "═") -> str:
        return left + m.join(ch * w for w in widths) + right

    title = " PIPELINE SUMMARY "
    total_width = sum(widths) + len(widths) + 1
    lines = [
        "╔" + "═" * (total_width - 2) + "╗",
        "║" + title.center(total_width - 2) + "║",
        _rule("╠", "╦", "╣"),
        _fmt(headers),
        _rule("╠", "╬", "╣"),
    ]
    for row in rows:
        lines.append(_fmt(row))
    lines += [
        _rule("╠", "╬", "╣"),
        _fmt(total_row),
        _rule("╚", "╩", "╝"),
    ]
    return "\n".join(lines)


def _render_repair_list(results: list[dict]) -> list[str]:
    lines: list[str] = []
    for r in results:
        for rep in r.get("repairs", []):
            cb = rep.get("cols_before", 0)
            ca = rep.get("cols_after", 0)
            cost = rep.get("cost", 0.0)
            lines.append(
                f"  → {rep['pdf_name']} page {rep['page_number'] + 1}, "
                f"table {rep['table_index'] + 1}: "
                f"columns {cb}→{ca}  (${cost:.5f})"
            )
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    load_dotenv(ROOT / ".env")

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set.\n"
            "Copy .env.example to .env and fill in your key, "
            "or export OPENAI_API_KEY in your shell.",
            file=sys.stderr,
        )
        return 1

    _ensure_dirs()
    logger = _setup_root_logger()

    model = os.environ.get("REPAIR_MODEL", "gpt-4o-mini")

    pdfs = _get_pdfs(args.pdf)
    if not pdfs:
        print(
            f"No PDFs found in {DATA_DIR}"
            + (f" matching '{args.pdf}'" if args.pdf else "")
            + ".\nPlace PDF files in the data/ folder and retry.",
            file=sys.stderr,
        )
        return 1

    from openai import OpenAI
    client = OpenAI()

    from pipeline import run_pipeline

    all_results: list[dict] = []
    all_success = True

    for pdf_path in pdfs:
        out_md = OUTPUT_DIR / f"{pdf_path.stem}.md"
        if out_md.exists() and not args.force:
            logger.info("SKIP (already exists — use --force): %s", pdf_path.name)
            all_results.append(
                {"pdf_name": pdf_path.name, "skipped": True, "repairs": []}
            )
            continue

        logger.info("Processing %s", pdf_path.name)
        try:
            result = run_pipeline(
                pdf_path=pdf_path,
                output_dir=OUTPUT_DIR,
                model=model,
                client=client,
                logger=logger,
                repairs_dir=REPAIRS_DIR,
            )
            all_results.append(result)
        except Exception:
            all_success = False
            tb = traceback.format_exc()
            logger.error("Pipeline FAILED for %s:\n%s", pdf_path.name, tb)
            all_results.append(
                {
                    "pdf_name": pdf_path.name,
                    "total_cost": 0.0,
                    "tables_found": 0,
                    "tables_validated": 0,
                    "tables_repaired": 0,
                    "tables_orphaned": 0,
                    "pages_processed": 0,
                    "repairs": [],
                    "error": tb,
                }
            )

    # Print summary
    processed = [r for r in all_results if not r.get("skipped")]
    print()
    if processed:
        print(_render_summary(processed))
        repair_lines = _render_repair_list(processed)
        if repair_lines:
            print("\nRepaired tables:")
            print("\n".join(repair_lines))
    print(f"\nRepair logs written to:  {REPAIRS_DIR}/")
    print(f"Output files written to: {OUTPUT_DIR}/")

    # Machine-readable summary
    summary_json = {
        "model": model,
        "results": [
            {k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()}
            for r in all_results
        ],
    }
    (LOG_DIR / "run_summary.json").write_text(
        json.dumps(summary_json, indent=2), encoding="utf-8"
    )

    # Tmp cleanup on full success
    if all_success:
        for item in TMP_DIR.glob("*"):
            try:
                item.unlink() if item.is_file() else shutil.rmtree(item)
            except Exception:
                pass

    # Optional self-verification test
    if args.test:
        print("\nRunning self-verification test...")
        rc = subprocess.call(
            [sys.executable, str(ROOT / "test_validation_case.py")],
            cwd=str(ROOT),
        )
        if rc != 0:
            all_success = False

    return 0 if all_success else 2


if __name__ == "__main__":
    sys.exit(main())
