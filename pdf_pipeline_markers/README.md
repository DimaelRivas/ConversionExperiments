# PDF → Markdown with Human-Review Markers

A single-purpose, **fully offline** pipeline (no LLM, no Camelot, no APIs, zero
running cost). It:

1. Extracts markdown from PDFs with **Docling** (text, headings, lists, tables)
2. **Scans Docling's markdown** for the two failure patterns it actually produces
   (see `scanner.py`) — no PDF geometry analysis
3. Inserts inline **HTML-comment review markers** wherever a table is suspect
4. Generates a **review checklist** per PDF telling a human exactly which tables to
   fix and on which source page

The goal is not to auto-fix tables — prior experiments showed automated repair can
make complex tables (e.g. competitor matrices with embedded product images) worse.
The goal is to make human review **fast and targeted**.

---

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** Docling downloads ML models (~1–2 GB) on first run. This requires an
> internet connection. Subsequent runs are fast and fully offline.

---

## 2. Usage

```bash
# Place PDFs in the data/ folder, then:

# Process all PDFs
python run.py

# Process a single PDF
python run.py --pdf test6.pdf

# Force reprocessing of already-completed files
python run.py --force

# Process and run the self-verification test
python run.py --pdf test6.pdf --test

# Run self-verification only (pipeline must have run first)
python test_validation_case.py
```

---

## 3. Output Files

```
output/
  test6.md              ← Full markdown with <!-- REVIEW NEEDED --> markers

logs/
  pipeline_run.log      ← Full execution log
  checklists/
    test6_checklist.md       ← All flagged tables for this PDF with instructions
    _master_checklist.md     ← All flagged tables across all PDFs
```

The output `.md` is a complete, readable document — every page of content is
present. Markers are HTML comments, so they are invisible when the markdown is
rendered but trivial to search for in a text editor.

---

## 4. Human Review Workflow

```
1. Open logs/checklists/_master_checklist.md
   → See all PDFs and how many tables need review at a glance

2. For each flagged table entry:
   → Open the output .md file
   → Search for  <!-- ⚠️ REVIEW NEEDED ... Page: N -->
   → Open the source PDF at page N
   → Manually correct the table below the marker
   → Delete the marker comment (and its <!-- END REVIEW SECTION -->) when done

3. When all markers are resolved:
   → The output .md file is ready for use
```

---

## 5. How Flagging Works

`scanner.py` reads Docling's markdown output and flags tables using two
heuristics — no PDF geometry analysis (PyMuPDF's `find_tables()` mis-reads
colored backgrounds in marketing PDFs as cells, producing false positives on
every correct table). The two patterns Docling actually exhibits:

- **`image_proximity`** — a table preceded (within 6 lines) by floating
  `<!-- image -->` elements, meaning its header row(s) contained images Docling
  could not turn into text. The checklist notes to replace image cells with text.
- **`split_table`** — one source table emitted as two consecutive table blocks
  with the same column count and nothing but blanks/images between them; both
  fragments are flagged so the reviewer merges them.

The scanner never overrides Docling's content — every decision is handed to a
human. Page numbers shown in the markers come from Docling's own JSON export
(not from any PDF geometry library), purely as a place for the reviewer to look.

---

## 6. Known Validation Case

`test6.pdf` contains two known problem tables:

- **"Resumen de Competencia"** — a wide competitor matrix with product images in
  cells, preceded by `<!-- image -->` elements → caught by `image_proximity`.
- **"Cuadro Comparativo Igualando"** — a table Docling splits into two
  same-column fragments → caught by `split_table`.

The clean market-data tables (*DDD Soles*, *DDD Unidades*, *Ejes Promocionales*)
must **not** be flagged — the self-verification test guards against exactly these
false positives. Verify automatically with:

```bash
python run.py --pdf test6.pdf --test
# or, after a run:
python test_validation_case.py
```
