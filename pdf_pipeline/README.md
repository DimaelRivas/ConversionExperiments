# PDF → Markdown Production Pipeline

**Architecture:** Docling extracts everything. PyMuPDF's `find_tables()` validates
table structure geometrically. The LLM (gpt-4o-mini) repairs only the tables that fail
validation, using a tightly cropped image — never a full page.

```
Docling  →  markdown + JSON metadata
              ↓
         PyMuPDF find_tables()  →  column/row counts per page
              ↓  (mismatch detected)
         gpt-4o-mini vision  →  cropped single-table image  →  repaired GFM table
              ↓
         clean_markdown()  →  output/{name}.md
```

---

## 1. System Requirements

```bash
# Python 3.10 or higher required
python --version
```

**First run:** Docling downloads ML models (~1–2 GB) on the first conversion.
This happens automatically but requires an internet connection.
Subsequent runs use the local cache and are fast.

---

## 2. Setup

```bash
# From the pdf_pipeline/ directory:
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
# Open .env and set your OPENAI_API_KEY
```

---

## 3. Usage

```bash
# Place your PDFs in data/ first, then:

# Process all PDFs in data/
python run.py

# Process a single PDF
python run.py --pdf test6.pdf

# Force re-processing (overwrite existing output)
python run.py --force

# Process and run the self-verification test
python run.py --pdf test6.pdf --test

# Run only the self-verification test (pipeline must have run first with test6.pdf)
python test_validation_case.py
```

---

## 4. Output Structure

```
output/
  test1.md              ← Final clean markdown
  test6.md

logs/
  pipeline_run.log      ← Full execution log (DEBUG level)
  run_summary.json      ← Machine-readable results
  repairs/
    test6_page11_table1.md   ← Diff report: original vs repaired
    test6_page11_table2.md
    _summary.md              ← All repairs at a glance
```

---

## 5. How to Review Results

Three artifacts per PDF:

1. **`output/{name}.md`** — final document, ready for human review
2. **`logs/repairs/_summary.md`** — all repaired tables in one overview table
3. **`logs/repairs/{name}_page{N}_table{M}.md`** — detailed side-by-side diff
   for each individual repair, including token usage and estimated cost

**Suggested review workflow:**
- Start with `_summary.md` to see which tables were touched
- Open individual diffs only where the column change looks unexpected
- The "Line-by-Line Diff" block uses standard unified-diff format:
  `-` lines were replaced, `+` lines are the LLM's reconstruction

---

## 6. Cost Model

The pipeline only calls the LLM for tables that fail structural validation.
On a typical 20-page document with 10 tables, expect 0–3 repairs per document.

| Model | Input | Output |
|---|---|---|
| gpt-4o-mini (default) | $0.15 / 1M tokens | $0.60 / 1M tokens |
| gpt-4o (override) | $2.50 / 1M tokens | $10.00 / 1M tokens |

Each table repair sends ~1,000–2,000 input tokens (image + prompt) and receives
~100–500 output tokens. Typical cost: **$0.0003–$0.0010 per repaired table**.

---

## 7. Known Validation Case

**test6.pdf, page 11** — two tables with two-tier headers:

```
| Marca | RECETAS MAT           ||| RECETAS TRIM          |||
|       | MAT 10/23 | MAT 10/24 | CREC% | TRIM 07/24 | TRIM 10/24 | CREC% |
```

Docling collapses these to 4 columns. PyMuPDF detects 7 columns per table.
Both tables are flagged and repaired. To verify automatically:

```bash
python run.py --pdf test6.pdf --test
# or
python test_validation_case.py
```

Expected output:
```
=== Self-Verification Test: test6.pdf Page 11 ===
[PASS] data/test6.pdf exists
[PASS] Pipeline completed without exception
[PASS] output/test6.md created and non-empty
[PASS] Repair log(s) found in logs/repairs/ for test6.pdf
[PASS] Column count changed in repair log: 4 → 7
[PASS] Repaired table near 'Receta por Especialidad' has more than 4 columns
=== Result: 6/6 checks passed ===
```
