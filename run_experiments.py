"""Orchestrator for the three PDF -> Markdown extraction pipelines.

Usage:
    python run_experiments.py [--pipeline a|b|c|all] [--pdf filename.pdf] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

import utils
from pipeline_a import run_pipeline_a
from pipeline_b import run_pipeline_b
from pipeline_c import run_pipeline_c

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
LOG_DIR = ROOT / "logs"
TMP_DIR = ROOT / "tmp"

PIPELINES = {
    "a": ("pipeline_a", "output/pipeline_a", run_pipeline_a),
    "b": ("pipeline_b", "output/pipeline_b", run_pipeline_b),
    "c": ("pipeline_c", "output/pipeline_c", run_pipeline_c),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PDF->Markdown pipelines.")
    parser.add_argument(
        "--pipeline", choices=["a", "b", "c", "all"], default="all",
        help="Which pipeline(s) to run (default: all)",
    )
    parser.add_argument(
        "--pdf", default=None,
        help="Process only this PDF (filename within data/). Default: all PDFs.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process even if an output .md already exists.",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    for d in (
        OUTPUT_DIR / "pipeline_a",
        OUTPUT_DIR / "pipeline_b",
        OUTPUT_DIR / "pipeline_c",
        LOG_DIR,
        TMP_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def get_client():
    from openai import OpenAI

    return OpenAI()


def select_pipelines(choice: str) -> list[str]:
    return ["a", "b", "c"] if choice == "all" else [choice]


def select_pdfs(pdf_arg: str | None) -> list[Path]:
    all_pdfs = utils.get_pdf_paths(str(DATA_DIR))
    if pdf_arg is None:
        return all_pdfs
    wanted = Path(pdf_arg).name
    matches = [p for p in all_pdfs if p.name == wanted]
    return matches


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------
def _cost_cell(entry: dict | None) -> str:
    if entry is None:
        return "-"
    status = entry["status"]
    if status == "ok":
        return f"${entry['cost']:.3f}"
    if status == "skipped":
        return "skip"
    return "ERR"


def render_summary(results: dict, pipelines: list[str]) -> str:
    headers = ["PDF", "PipelineA", "PipelineB", "PipelineC", "Total Cost"]
    rows = []
    totals = {"a": 0.0, "b": 0.0, "c": 0.0}
    grand_total = 0.0

    for pdf_name in sorted(results):
        per = results[pdf_name]
        row_total = 0.0
        cells = [pdf_name]
        for key in ("a", "b", "c"):
            entry = per.get(key)
            cells.append(_cost_cell(entry))
            if entry and entry["status"] == "ok":
                totals[key] += entry["cost"]
                row_total += entry["cost"]
        grand_total += row_total
        cells.append(f"${row_total:.3f}")
        rows.append(cells)

    total_row = [
        "TOTAL",
        f"${totals['a']:.3f}",
        f"${totals['b']:.3f}",
        f"${totals['c']:.3f}",
        f"${grand_total:.3f}",
    ]

    # Column widths
    all_rows = [headers] + rows + [total_row]
    widths = [max(len(r[i]) for r in all_rows) + 2 for i in range(len(headers))]

    def fmt(cells, sep="║"):
        return sep + sep.join(c.center(widths[i]) for i, c in enumerate(cells)) + sep

    def rule(left, mid, right, ch="─"):
        return left + mid.join(ch * w for w in widths) + right

    lines = []
    title = " EXPERIMENT SUMMARY "
    full_width = sum(widths) + len(widths) + 1
    lines.append("╔" + "═" * (full_width - 2) + "╗")
    lines.append("║" + title.center(full_width - 2) + "║")
    lines.append(rule("╠", "╦", "╣", "═"))
    lines.append(fmt(headers))
    lines.append(rule("╠", "╬", "╣", "═"))
    for r in rows:
        lines.append(fmt(r))
    lines.append(rule("╠", "╬", "╣", "═"))
    lines.append(fmt(total_row))
    lines.append(rule("╚", "╩", "╝", "═"))
    return "\n".join(lines)


def render_stats(results: dict) -> list[str]:
    """Aggregate pipeline-specific stats lines."""
    a_found = a_flagged = a_repaired = 0
    c_found = c_camelot = c_fallback = c_failed = 0
    for per in results.values():
        a = per.get("a")
        if a and a["status"] == "ok" and a["result"]:
            r = a["result"]
            a_found += r.get("tables_found", 0)
            a_flagged += r.get("tables_flagged", 0)
            a_repaired += r.get("tables_repaired", 0)
        c = per.get("c")
        if c and c["status"] == "ok" and c["result"]:
            r = c["result"]
            cam = r.get("tables_camelot_success", 0)
            fb = r.get("tables_llm_fallback", 0)
            fl = r.get("tables_failed", 0)
            c_camelot += cam
            c_fallback += fb
            c_failed += fl
            c_found += cam + fb + fl

    lines = []
    lines.append(
        f"Pipeline A: {a_found} tables found, {a_flagged} flagged, {a_repaired} repaired"
    )
    lines.append(
        f"Pipeline C: {c_found} tables found, {c_camelot} via Camelot, "
        f"{c_fallback} via LLM fallback, {c_failed} failed"
    )
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set. Copy .env.example to .env and set "
            "your key, or export OPENAI_API_KEY in your shell.",
            file=sys.stderr,
        )
        return 1

    ensure_dirs()

    model_b = os.environ.get("PIPELINE_B_MODEL", "gpt-4o")
    model_ac = os.environ.get("PIPELINE_AC_MODEL", "gpt-4o-mini")

    pipelines = select_pipelines(args.pipeline)
    pdfs = select_pdfs(args.pdf)

    if not pdfs:
        where = f"matching '{args.pdf}' " if args.pdf else ""
        print(f"No PDFs {where}found in {DATA_DIR}. Place PDFs there and retry.")
        return 1

    client = get_client()

    results: dict[str, dict] = {}
    all_success = True

    for pidx, pdf_path in enumerate(pdfs, start=1):
        results.setdefault(pdf_path.name, {})
        for key in pipelines:
            pipeline_name, out_subdir, fn = PIPELINES[key]
            out_dir = ROOT / out_subdir
            out_file = out_dir / f"{pdf_path.stem}.md"
            label = f"Pipeline {key.upper()}"
            print(
                f"[{label}] Processing {pdf_path.name} "
                f"({pidx}/{len(pdfs)})..."
            )

            # Idempotency
            if out_file.exists() and not args.force:
                print(f"  -> exists, skipping (use --force to re-run): {out_file}")
                results[pdf_path.name][key] = {
                    "status": "skipped", "cost": 0.0, "result": None, "error": None,
                }
                continue

            model = model_b if key == "b" else model_ac
            logger = utils.setup_logging(pipeline_name, pdf_path.name, LOG_DIR)
            try:
                res = fn(pdf_path, out_dir, model, client, logger)
                results[pdf_path.name][key] = {
                    "status": "ok",
                    "cost": float(res.get("total_cost", 0.0)),
                    "result": res,
                    "error": None,
                }
            except Exception:  # noqa: BLE001
                all_success = False
                tb = traceback.format_exc()
                logger.error("Pipeline %s failed on %s:\n%s", key, pdf_path.name, tb)
                print(f"  -> FAILED: see logs/{pipeline_name}_{pdf_path.stem}.log")
                results[pdf_path.name][key] = {
                    "status": "error", "cost": 0.0, "result": None, "error": tb,
                }

    # --- Summary --------------------------------------------------------
    print()
    print(render_summary(results, pipelines))
    print()
    for line in render_stats(results):
        print(line)
    print()
    print("Output files written to:")
    for key in pipelines:
        print(f"  {PIPELINES[key][1]}/")

    # --- Machine-readable summary --------------------------------------
    summary_json = {
        "models": {"pipeline_b": model_b, "pipeline_ac": model_ac},
        "pipelines_run": pipelines,
        "results": {
            pdf: {
                k: {
                    "status": v["status"],
                    "cost": v["cost"],
                    "result": v["result"],
                    "error": v["error"],
                }
                for k, v in per.items()
            }
            for pdf, per in results.items()
        },
    }
    summary_path = LOG_DIR / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(f"\nMachine-readable summary: {summary_path}")

    # --- Tmp cleanup (only if everything succeeded) ---------------------
    if all_success:
        for f in TMP_DIR.glob("*"):
            try:
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
            except Exception:  # noqa: BLE001
                pass
        print("Temporary files cleaned.")
    else:
        print("Some pipelines failed; leaving tmp/ in place for inspection.")

    return 0 if all_success else 2


if __name__ == "__main__":
    sys.exit(main())
