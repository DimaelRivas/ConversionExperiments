# Docling PDF Ablation Framework

This project runs the same PDF set through multiple Docling extraction profiles and writes comparable Markdown, JSON, tables, images, metadata, and summary reports.

## Install Locally

```bash
python -m venv .venv
source .venv/bin/activate
bash scripts/install_ablation_env.sh
```

The local machine does not need a GPU. The installer checks Python, Docling import/version, PyTorch CUDA availability if PyTorch is installed, and `nvidia-smi` if present.

Tesseract OCR profiles also need system packages. On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa
```

EasyOCR downloads model weights on first use. GPU/VLM profiles may require extra model downloads depending on the installed Docling version and runtime.

## Copy To Remote GPU Server

Copy the full project directory, including `configs/`, `scripts/`, `requirements.txt`, and your PDFs:

```bash
rsync -av --exclude .venv ./ user@gpu-server:/path/ConversionExperiments/
```

Then on the remote server:

```bash
cd /path/ConversionExperiments
python -m venv .venv
source .venv/bin/activate
bash scripts/install_ablation_env.sh
```

Confirm `nvidia-smi` reports a GPU before running CUDA or VLM profiles.

## Run CPU-Safe Profiles

Put PDFs under `data/`, then run all profiles. GPU-required profiles are recorded as skipped unless `--allow-gpu` is passed.

```bash
python scripts/run_docling_ablation.py \
  --input data \
  --profiles all \
  --output outputs/docling_ablation
```

Run selected CPU-safe profiles:

```bash
python scripts/run_docling_ablation.py \
  --input data \
  --profiles standard_fast_no_ocr,standard_accurate_no_ocr \
  --output outputs/docling_ablation
```

## Run GPU Profiles

GPU-required profiles never run accidentally. Pass `--allow-gpu` to enable them:

```bash
python scripts/run_docling_ablation.py \
  --input data/FICHA_TECNICA_TALFLEX_BI_LIB_PROLONG.pdf \
  --profiles standard_accurate_no_ocr,standard_accurate_cellmatch_false,vlm_granite_docling_cuda \
  --output outputs/docling_ablation \
  --allow-gpu
```

If a CUDA/VLM profile is selected and no GPU is found, the runner fails clearly. To record a skip instead:

```bash
python scripts/run_docling_ablation.py \
  --input data \
  --profiles vlm_granite_docling_cuda \
  --output outputs/docling_ablation \
  --allow-gpu \
  --allow-missing-gpu
```

Use `--device-override cpu`, `--device-override auto`, or `--device-override cuda` when you need to force device selection for all selected profiles.

## Output Layout

Each PDF/profile produces:

```text
outputs/docling_ablation/
└── <pdf_stem>/
    └── <profile_name>/
        ├── output.md
        ├── output.json
        ├── profile_used.yaml
        ├── run_metadata.json
        ├── tables/
        │   ├── table_001.csv
        │   ├── table_001.html
        │   └── ...
        ├── images/
        │   └── ...
        └── errors.log
```

After every run, the root output folder also contains:

- `summary.csv`
- `summary.xlsx`
- `summary.md`

The summary has one row per PDF/profile with status, runtime, device, GPU details, table count, output paths, and error or skip reason.

## Profile Guide

- `standard_fast_no_ocr`: fast baseline for native text PDFs.
- `standard_accurate_no_ocr`: native text PDFs where table quality matters more than runtime.
- `standard_accurate_cellmatch_false`: difficult tables or merged-column issues; compare against cell matching enabled.
- `standard_tesseract_ocr_psm3`: scanned or mixed PDFs using Tesseract with general page segmentation.
- `standard_tesseract_force_ocr_psm6`: bad embedded text layers where full-page OCR is needed.
- `standard_easyocr_force_ocr`: image-heavy PDFs where EasyOCR may outperform Tesseract.
- `standard_cuda_accurate`: standard pipeline with explicit CUDA acceleration.
- `vlm_granite_docling_cuda`: VLM visual extraction with Granite Docling; practical only on GPU.

Markdown is convenient for inspection but can be lossy. Treat `output.json` as the source of truth when comparing structure, tables, captions, pictures, and layout details.

## Docling Compatibility Notes

The runner uses the Docling Python API with `PdfPipelineOptions`, `DocumentConverter`, `PdfFormatOption`, and `InputFormat.PDF`. It maps YAML table options to `TableFormerMode.FAST` or `TableFormerMode.ACCURATE`, sets `table_structure_options.do_cell_matching`, and configures OCR, image generation, and accelerator options when those fields exist.

Docling has changed some OCR and VLM class names across versions. The script tries the common Tesseract, EasyOCR, accelerator, and VLM option classes and raises a clear profile-level error if the installed Docling version cannot support a selected option. Table export is also defensive: it tries table/dataframe/html export methods and records failures in `errors.log` without stopping the whole batch.

## Useful Flags

- `--overwrite`: rerun profiles even when previous metadata exists.
- `--max-pages N`: quick test on the first N pages when supported by the installed Docling converter.
- `--allow-missing-gpu`: record CUDA/VLM profiles as skipped on CPU-only machines.
- `--config`: point to a modified profile YAML.

One failed profile does not stop the batch. The profile is marked failed, `errors.log` gets a traceback, and summary files are still generated.
