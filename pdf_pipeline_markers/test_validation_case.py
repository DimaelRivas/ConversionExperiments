"""Self-verification for the markdown scanner against test6.pdf.

Run standalone (the pipeline must have already produced output/test6.md):
    python test_validation_case.py

Validates the two heuristics against the known patterns in test6.pdf, and — most
importantly — that the clean market-data tables are NOT flagged (these were the
false positives the old PyMuPDF validator produced).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR   = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


def check_image_proximity(output_text: str) -> tuple[bool, str]:
    """Resumen de Competencia table should be flagged."""
    lines = output_text.splitlines()
    for i, line in enumerate(lines):
        if "Resumen de Competencia" in line:
            window = "\n".join(lines[i : i + 15])
            if "REVIEW NEEDED" in window and ("image_proximity" in window.lower() or "image" in window.lower()):
                return True, "Marker found near Resumen de Competencia"
            if "REVIEW NEEDED" in window:
                return True, "Marker found near Resumen de Competencia"
    return False, "No marker found near Resumen de Competencia"


def check_split_table(output_text: str) -> tuple[bool, str]:
    """Split table fragments should both be flagged."""
    lines = output_text.splitlines()
    for i, line in enumerate(lines):
        if "Igualando" in line:
            window = "\n".join(lines[i : i + 40])
            count = window.count("REVIEW NEEDED")
            if count >= 2:
                return True, f"Found {count} markers in Igualando section (expected ≥2 for split fragments)"
            if count == 1:
                return True, "Found 1 marker in Igualando section (partial — both fragments should be flagged)"
    return False, "No marker found in Igualando section"


def check_no_false_positives(output_text: str) -> tuple[bool, str]:
    """Clean tables must not be flagged."""
    lines = output_text.splitlines()
    false_positives = []
    clean_sections = ["DDD Soles", "DDD Unidades de cuenta", "Ejes Promocionales"]

    for section in clean_sections:
        for i, line in enumerate(lines):
            if section in line and "##" in line:
                window = "\n".join(lines[i : i + 5])
                if "REVIEW NEEDED" in window:
                    false_positives.append(section)
                break

    if false_positives:
        return False, f"FALSE POSITIVES detected on: {', '.join(false_positives)}"
    return True, "No false positives on clean tables"


def main() -> int:
    print("=== Self-Verification: test6.pdf Markdown Scanner ===")
    pdf_path = DATA_DIR / "test6.pdf"
    out_md = OUTPUT_DIR / "test6.md"

    passed = total = 0

    # Check: PDF exists
    total += 1
    if pdf_path.exists():
        print("[PASS] data/test6.pdf exists")
        passed += 1
    else:
        print(f"[SKIP] data/test6.pdf not found in {DATA_DIR} — place the file there and re-run")
        print("=== Result: skipped ===")
        return 0

    # Check: output exists
    total += 1
    if out_md.exists() and out_md.stat().st_size > 0:
        print(f"[PASS] output/test6.md exists ({out_md.stat().st_size:,} bytes)")
        passed += 1
    else:
        print("[FAIL] output/test6.md missing or empty — run the pipeline first")
        print(f"\n=== Result: {passed}/{total} checks passed ===")
        return 1

    output_text = out_md.read_text(encoding="utf-8")

    # Check 1: image proximity
    total += 1
    ok, detail = check_image_proximity(output_text)
    print(f"[{'PASS' if ok else 'FAIL'}] Image proximity heuristic — Resumen de Competencia flagged correctly\n         {detail}")
    if ok:
        passed += 1

    # Check 2: split table
    total += 1
    ok, detail = check_split_table(output_text)
    print(f"[{'PASS' if ok else 'FAIL'}] Split table heuristic — Igualando section has marker(s)\n         {detail}")
    if ok:
        passed += 1

    # Check 3: no false positives (critical)
    total += 1
    ok, detail = check_no_false_positives(output_text)
    print(f"[{'PASS' if ok else 'FAIL'}] No false positives — DDD Soles, DDD Unidades, Ejes Promocionales are clean\n         {detail}")
    if ok:
        passed += 1
    elif not ok:
        # Make the critical failure unmissable
        print(f"         >>> CHECK 3 FAILED: {detail}")

    print(f"\n=== Result: {passed}/{total} checks passed ===")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
