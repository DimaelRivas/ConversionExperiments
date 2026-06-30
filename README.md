# PDF → Markdown Extraction — Three-Pipeline Experiment

Three independent strategies convert every PDF in `data/` into Markdown, so their
output (and cost) can be compared before a human review step. Each pipeline writes
one `<filename>.md` per PDF and the run ends with a token/cost summary.

| Pipeline | Strategy | Model |
|---|---|---|
| **A** | Docling base extraction + targeted **LLM repair** of low-quality tables | `gpt-4o-mini` |
| **B** | **Full-page vision** conversion of every page (reference / ground truth) | `gpt-4o` |
| **C** | PyMuPDF layout + **Camelot** tables (lattice→stream) + **LLM fallback** | `gpt-4o-mini` |

---

## 1. System dependencies (OS-level, before pip)

Camelot requires **Ghostscript** installed at the OS level.

```bash
# macOS
brew install ghostscript

# Ubuntu/Debian
sudo apt-get install ghostscript

# Verify
gs --version
```

## 2. Python environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Environment configuration

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

Configurable variables (`.env`):

| Variable | Default | Used by |
|---|---|---|
| `OPENAI_API_KEY` | — (required) | all LLM calls |
| `PIPELINE_B_MODEL` | `gpt-4o` | Pipeline B |
| `PIPELINE_AC_MODEL` | `gpt-4o-mini` | Pipelines A & C |

## 4. Running the experiments

```bash
# Place your PDFs in the data/ folder first

# Run all three pipelines on all PDFs
python run_experiments.py

# Run only Pipeline B (ground truth) on all PDFs
python run_experiments.py --pipeline b

# Run all pipelines on a specific PDF
python run_experiments.py --pdf test1.pdf

# Run Pipeline A and C (skip the expensive B)
python run_experiments.py --pipeline a
python run_experiments.py --pipeline c

# Re-process even if an output .md already exists (default is to skip)
python run_experiments.py --force
```

Runs are **idempotent**: if an output file already exists it is skipped and logged.
Use `--force` to override.

## 5. Output location

```
output/
├── pipeline_a/test1.md   ← Docling + LLM table repair
├── pipeline_b/test1.md   ← Full page vision (ground truth)
└── pipeline_c/test1.md   ← Camelot + LLM fallback
```

Per-pipeline, per-PDF logs are written to `logs/`, and a machine-readable summary
to `logs/experiment_summary.json`.

## 6. First run note

Docling downloads ML models on first use (~1–2 GB). This happens automatically but
requires an internet connection. Subsequent runs are fast.

---

## How table handling differs

- **Pipeline A** trusts Docling, then *scores* each extracted markdown table with
  heuristics (column consistency, empty-cell ratio, truncation markers, row count,
  encoding artifacts, cell length). A table is flagged when ≥ 2 issues appear and is
  re-extracted from the rendered page image by the LLM.
- **Pipeline B** sends every page image to a high-quality vision model and stitches
  paragraphs/tables split across page boundaries with regex-only logic.
- **Pipeline C** isolates each table region detected by PyMuPDF, tries Camelot
  `lattice` (accuracy ≥ 80) then `stream` (accuracy ≥ 70), and only falls back to the
  LLM on the cropped region when both fail.

## Cost tracking

Every OpenAI call returns its token usage, which is priced with the table in
`utils.py` (`PRICING`) and accumulated per pipeline. The final summary prints real
USD costs per PDF and a grand total.
