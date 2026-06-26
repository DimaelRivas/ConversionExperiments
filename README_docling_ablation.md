# Docling PDF Table Ablation Suite

This repository runs the same PDF or PDF folder through multiple Docling conversion profiles and writes comparable Markdown, JSON, table CSV/HTML files, images, metadata, inspections, and summary reports. The current suite is tuned for maximum table extraction quality, not minimum runtime.

The newer profiles are intentionally heavier. They vary TableFormer mode, cell matching, OCR strategy, OCR engine, Tesseract PSM, PDF backend, device, VLM pipeline, image scale, and batch/performance options. CPU-only development remains supported, while CUDA and VLM profiles are gated behind `--allow-gpu` for remote NVIDIA runs.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
bash scripts/install_ablation_env.sh
```

If `python` is not available, use `python3` for the venv command. The installer checks Python, imports, Docling version, Tesseract, `nvidia-smi`, and Torch CUDA when available.

Tesseract OCR profiles need system packages. On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa
```

## Common Runs

List all configured profiles:

```bash
python scripts/run_docling_ablation.py --list-profiles
```

CPU-safe strong profiles:

```bash
python scripts/run_docling_ablation.py \
  --input data \
  --profile-groups native_text,ocr,backend \
  --quality-tiers strong,heavy \
  --output outputs/docling_ablation
```

Heavy GPU run:

```bash
python scripts/run_docling_ablation.py \
  --input data \
  --quality-tiers heavy,extreme \
  --resource-classes high,extreme \
  --output outputs/docling_ablation \
  --allow-gpu
```

VLM-only GPU run:

```bash
python scripts/run_docling_ablation.py \
  --input data \
  --profile-groups vlm \
  --output outputs/docling_ablation \
  --allow-gpu
```

Dry run:

```bash
python scripts/run_docling_ablation.py \
  --input data \
  --quality-tiers extreme \
  --allow-gpu \
  --dry-run-plan
```

The legacy command still works:

```bash
python scripts/run_docling_ablation.py \
  --input data \
  --profiles all \
  --output outputs/docling_ablation \
  --allow-gpu
```

## Profile Families

- Native-text PDFs: `native_text` profiles compare accurate TableFormer extraction across `docling_parse`, `pypdfium2`, threaded parsing, image scale, and cell matching.
- Scanned PDFs: forced OCR profiles use Tesseract, EasyOCR, or RapidOCR where supported.
- Bad embedded text layers: forced full-page OCR profiles replace unreliable selectable text.
- Difficult tables: heavy and extreme profiles increase image scale and combine OCR, backend, and CUDA settings.
- Merged-column issues: `cell_matching` profiles disable cell matching to test whether Docling is over-merging columns.
- Image-heavy PDFs: EasyOCR and high image-scale profiles are intended for visually complex documents.
- VLM visual extraction: `vlm` profiles use Docling VLM presets such as Granite Docling, SmolDocling, or Qwen when supported by the installed Docling version.

Unsupported RapidOCR or VLM APIs are skipped cleanly and recorded in `run_metadata.json`; one unsupported profile does not stop the batch.

## Output Layout

```text
outputs/docling_ablation/
  summary.csv
  summary.xlsx
  summary.md
  table_quality_summary.csv
  table_quality_summary.xlsx
  table_quality_summary.md
  best_candidates.csv
  best_candidates.xlsx
  best_candidates.md
  <pdf_stem>/
    <profile_name>/
      output.md
      output.json
      profile_used.yaml
      run_metadata.json
      table_metrics.json
      inspection.md
      tables/
        table_001.csv
        table_001.html
      images/
      errors.log
```

`output.md` is convenient for quick reading, but Markdown is lossy. Use `output.json` as the source of truth for structure, table items, captions, page references, pictures, and layout details.

`tables/*.csv` and `tables/*.html` are best-effort exports from Docling table APIs. If direct export is unavailable, the runner tries table-like document items and records `table_export_failed` without failing the conversion.

`table_metrics.json` contains ranking signals such as table count, row/column totals, empty-cell ratio, single-column table count, very-small table count, Markdown table-line count, and JSON table item count. These are not ground-truth quality metrics.

`inspection.md` is the fastest per-run review file. It lists the profile, runtime, status, table CSV paths, top table shapes, the first 20 Markdown lines, and any errors or skip reasons.

`summary.*` gives one row per PDF/profile with status, runtime, device/GPU details, selected settings, and output paths.

`table_quality_summary.*` focuses on table signals across profiles.

`best_candidates.*` ranks successful runs heuristically. The ranking is only a triage aid; inspect JSON and CSV/HTML outputs manually before choosing a final extraction profile.

## GPU Workflow

Develop locally with CPU-safe filters and dry-runs, then copy the repository and PDFs to the GPU server:

```bash
rsync -av --exclude .venv ./ user@gpu-server:/path/ConversionExperiments/
```

On the server:

```bash
cd /path/ConversionExperiments
python -m venv .venv
source .venv/bin/activate
bash scripts/install_ablation_env.sh
nvidia-smi
```

Then run CUDA/VLM profiles with `--allow-gpu`. If you want missing GPUs recorded as skips instead of failures, also pass `--allow-missing-gpu`.

## Useful Flags

- `--profiles all` or `--profiles name1,name2`: select exact profiles.
- `--profile-groups native_text,ocr,cell_matching,vlm,gpu_performance,heavy_quality`: filter by metadata group.
- `--quality-tiers strong,heavy,extreme`: filter by quality tier.
- `--resource-classes high,extreme`: filter by expected resource use.
- `--list-profiles`: print profile metadata and exit.
- `--dry-run-plan`: print PDF/profile run and skip decisions without importing Docling.
- `--max-pages N`: use Docling page limiting when available; otherwise a warning is recorded.
- `--device-override cpu|auto|cuda`: override profile device selection.
- `--overwrite`: rerun profiles even when previous metadata exists.

One failed profile does not stop the batch. Failed and skipped profiles still produce `run_metadata.json`, `errors.log`, and `inspection.md`, and root summary files are still generated.
