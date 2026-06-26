#!/usr/bin/env python3
"""Run reproducible Docling extraction ablations over one PDF or a PDF folder."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("docling_ablation")


class SimpleProgress:
    """Minimal tqdm-compatible fallback used before optional deps are installed."""

    def __init__(self, total: int, desc: str = "", unit: str = "") -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.current = 0

    def __enter__(self) -> "SimpleProgress":
        if self.total:
            LOGGER.info("%s: 0/%s %s", self.desc, self.total, self.unit)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.total:
            LOGGER.info("%s: %s/%s %s", self.desc, self.current, self.total, self.unit)

    def update(self, count: int = 1) -> None:
        self.current += count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Docling PDF extraction ablation profiles and save structured outputs."
    )
    parser.add_argument("--input", required=True, help="PDF file or folder containing PDFs.")
    parser.add_argument("--profiles", required=True, help="'all' or comma-separated profile names.")
    parser.add_argument("--config", default="configs/docling_ablation_profiles.yaml")
    parser.add_argument("--output", default="outputs/docling_ablation")
    parser.add_argument("--allow-gpu", action="store_true", help="Enable GPU-required profiles.")
    parser.add_argument(
        "--allow-missing-gpu",
        action="store_true",
        help="Skip selected GPU-required profiles when no GPU is available instead of failing.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rerun even if outputs already exist.")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional page limit for quick tests.")
    parser.add_argument(
        "--device-override",
        choices=["cpu", "auto", "cuda"],
        default=None,
        help="Override profile device selection.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_profiles(config_path: Path) -> dict[str, dict[str, Any]]:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"No 'profiles' mapping found in {config_path}")
    return profiles


def select_profiles(all_profiles: dict[str, dict[str, Any]], spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(all_profiles.keys())
    requested = [item.strip() for item in spec.split(",") if item.strip()]
    missing = [name for name in requested if name not in all_profiles]
    if missing:
        raise ValueError(f"Unknown profile(s): {', '.join(missing)}")
    return requested


def find_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {input_path}")
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*.pdf") if path.is_file())
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def run_command(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
        text = (result.stdout or result.stderr or "").strip()
        return result.returncode == 0, text
    except Exception as exc:
        return False, str(exc)


def detect_gpu() -> dict[str, Any]:
    info: dict[str, Any] = {
        "available": False,
        "name": None,
        "memory_total_mb": None,
        "memory_used_mb": None,
        "sources": [],
    }

    if shutil.which("nvidia-smi"):
        ok, output = run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        if ok and output:
            first = output.splitlines()[0]
            parts = [part.strip() for part in first.split(",")]
            info["available"] = True
            info["sources"].append("nvidia-smi")
            if parts:
                info["name"] = parts[0]
            if len(parts) >= 3:
                info["memory_total_mb"] = _safe_int(parts[1])
                info["memory_used_mb"] = _safe_int(parts[2])

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            info["available"] = True
            info["sources"].append("torch")
            info["name"] = info["name"] or torch.cuda.get_device_name(0)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                info["memory_total_mb"] = info["memory_total_mb"] or int(total_bytes / 1024 / 1024)
                info["memory_used_mb"] = int((total_bytes - free_bytes) / 1024 / 1024)
            except Exception:
                pass
    except Exception:
        pass

    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info["available"] = True
            info["sources"].append("pynvml")
            info["name"] = info["name"] or name
            info["memory_total_mb"] = info["memory_total_mb"] or int(memory.total / 1024 / 1024)
            info["memory_used_mb"] = int(memory.used / 1024 / 1024)
        pynvml.nvmlShutdown()
    except Exception:
        pass

    info["sources"] = sorted(set(info["sources"]))
    return info


def _safe_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def docling_version() -> str:
    try:
        import docling  # type: ignore

        return str(getattr(docling, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def import_docling_api() -> dict[str, Any]:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except Exception as exc:
        raise RuntimeError(
            "Could not import Docling Python API. Install dependencies with "
            "scripts/install_ablation_env.sh or python -m pip install -r requirements.txt. "
            f"Original error: {exc}"
        ) from exc

    api: dict[str, Any] = {
        "InputFormat": InputFormat,
        "PdfPipelineOptions": PdfPipelineOptions,
        "TableFormerMode": TableFormerMode,
        "DocumentConverter": DocumentConverter,
        "PdfFormatOption": PdfFormatOption,
    }

    for name in [
        "EasyOcrOptions",
        "TesseractCliOcrOptions",
        "TesseractOcrOptions",
        "AcceleratorOptions",
        "AcceleratorDevice",
        "VlmPipelineOptions",
        "GraniteDoclingVlmOptions",
    ]:
        try:
            module = __import__("docling.datamodel.pipeline_options", fromlist=[name])
            api[name] = getattr(module, name)
        except Exception:
            api[name] = None

    try:
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

        api["StandardPdfPipeline"] = StandardPdfPipeline
    except Exception:
        api["StandardPdfPipeline"] = None

    try:
        from docling.pipeline.vlm_pipeline import VlmPipeline

        api["VlmPipeline"] = VlmPipeline
    except Exception:
        api["VlmPipeline"] = None

    return api


def build_pipeline_options(profile: dict[str, Any], selected_device: str, api: dict[str, Any]) -> Any:
    pipeline_type = profile.get("pipeline", "standard")
    options_cls = api["PdfPipelineOptions"]
    if pipeline_type == "vlm" and api.get("VlmPipelineOptions") is not None:
        options_cls = api["VlmPipelineOptions"]
    elif pipeline_type == "vlm" and api.get("VlmPipelineOptions") is None:
        raise RuntimeError(
            "This Docling version does not expose VlmPipelineOptions. "
            "Upgrade Docling or run a standard profile."
        )

    options = options_cls()
    set_if_present(options, "do_ocr", bool(profile.get("do_ocr", False)))
    set_if_present(options, "force_full_page_ocr", bool(profile.get("force_full_page_ocr", False)))
    set_if_present(options, "do_table_structure", bool(profile.get("do_table_structure", True)))
    set_if_present(options, "generate_page_images", bool(profile.get("generate_page_images", False)))
    set_if_present(options, "generate_picture_images", bool(profile.get("generate_picture_images", False)))
    set_if_present(options, "images_scale", float(profile.get("images_scale", 1.0)))

    table_options = getattr(options, "table_structure_options", None)
    if table_options is not None:
        set_if_present(table_options, "do_cell_matching", bool(profile.get("do_cell_matching", True)))
        mode = table_mode(profile.get("table_mode", "accurate"), api["TableFormerMode"])
        set_if_present(table_options, "mode", mode)

    if bool(profile.get("do_ocr", False)):
        options.ocr_options = build_ocr_options(profile, api)

    accelerator = build_accelerator_options(selected_device, api)
    if accelerator is not None:
        set_if_present(options, "accelerator_options", accelerator)

    if pipeline_type == "vlm":
        configure_vlm_options(options, profile, api)

    return options


def set_if_present(obj: Any, attr: str, value: Any) -> None:
    if hasattr(obj, attr):
        setattr(obj, attr, value)


def table_mode(mode_name: str, enum_cls: Any) -> Any:
    normalized = str(mode_name).strip().lower()
    if normalized == "fast":
        return getattr(enum_cls, "FAST")
    if normalized == "accurate":
        return getattr(enum_cls, "ACCURATE")
    raise ValueError(f"Unsupported table_mode: {mode_name}")


def build_ocr_options(profile: dict[str, Any], api: dict[str, Any]) -> Any:
    engine = str(profile.get("ocr_engine", "easyocr")).lower()
    languages = list(profile.get("ocr_lang") or [])
    psm = profile.get("psm")

    if engine == "tesseract":
        option_cls = api.get("TesseractCliOcrOptions") or api.get("TesseractOcrOptions")
        if option_cls is None:
            raise RuntimeError(
                "This Docling version does not expose Tesseract OCR options. "
                "Install a newer Docling version or disable Tesseract profiles."
            )
        kwargs = compatible_kwargs(option_cls, {"lang": languages, "psm": psm})
        if "lang" not in kwargs:
            kwargs.update(compatible_kwargs(option_cls, {"languages": languages}))
        return option_cls(**{k: v for k, v in kwargs.items() if v is not None})

    if engine == "easyocr":
        option_cls = api.get("EasyOcrOptions")
        if option_cls is None:
            raise RuntimeError(
                "This Docling version does not expose EasyOcrOptions. "
                "Install a newer Docling version or disable EasyOCR profiles."
            )
        kwargs = compatible_kwargs(option_cls, {"lang": languages})
        if "lang" not in kwargs:
            kwargs.update(compatible_kwargs(option_cls, {"languages": languages}))
        return option_cls(**kwargs)

    raise ValueError(f"Unsupported OCR engine: {engine}")


def compatible_kwargs(cls: Any, values: dict[str, Any]) -> dict[str, Any]:
    try:
        params = inspect.signature(cls).parameters
        return {key: value for key, value in values.items() if key in params}
    except Exception:
        return values


def build_accelerator_options(device: str, api: dict[str, Any]) -> Any | None:
    accelerator_cls = api.get("AcceleratorOptions")
    if accelerator_cls is None:
        return None

    accelerator_device = api.get("AcceleratorDevice")
    value: Any = device
    if accelerator_device is not None:
        lookup = {"cpu": "CPU", "cuda": "CUDA", "auto": "AUTO"}
        enum_name = lookup.get(device, "AUTO")
        value = getattr(accelerator_device, enum_name, device)

    try:
        return accelerator_cls(device=value)
    except Exception:
        obj = accelerator_cls()
        set_if_present(obj, "device", value)
        return obj


def configure_vlm_options(options: Any, profile: dict[str, Any], api: dict[str, Any]) -> None:
    model_name = str(profile.get("vlm_model", "")).lower()
    if model_name != "granite_docling":
        return
    granite_cls = api.get("GraniteDoclingVlmOptions")
    if granite_cls is None:
        LOGGER.warning("GraniteDoclingVlmOptions not found; using Docling VLM defaults.")
        return
    try:
        set_if_present(options, "vlm_options", granite_cls())
    except Exception as exc:
        raise RuntimeError(f"Could not configure Granite Docling VLM options: {exc}") from exc


def build_converter(profile: dict[str, Any], pipeline_options: Any, api: dict[str, Any]) -> Any:
    DocumentConverter = api["DocumentConverter"]
    PdfFormatOption = api["PdfFormatOption"]
    InputFormat = api["InputFormat"]

    kwargs: dict[str, Any] = {"pipeline_options": pipeline_options}
    if profile.get("pipeline") == "vlm":
        if api.get("VlmPipeline") is None:
            raise RuntimeError(
                "This Docling version does not expose VlmPipeline. "
                "Upgrade Docling or remove VLM profiles."
            )
        kwargs["pipeline_cls"] = api["VlmPipeline"]
    elif api.get("StandardPdfPipeline") is not None:
        kwargs["pipeline_cls"] = api["StandardPdfPipeline"]

    try:
        pdf_option = PdfFormatOption(**kwargs)
    except TypeError:
        kwargs.pop("pipeline_cls", None)
        pdf_option = PdfFormatOption(**kwargs)

    return DocumentConverter(format_options={InputFormat.PDF: pdf_option})


def convert_pdf(converter: Any, pdf_path: Path, max_pages: int | None) -> Any:
    try:
        if max_pages is not None and "max_num_pages" in inspect.signature(converter.convert).parameters:
            return converter.convert(pdf_path, max_num_pages=max_pages)
    except Exception:
        pass
    return converter.convert(pdf_path)


def export_document(doc: Any, run_dir: Path) -> dict[str, str | None]:
    md_path = run_dir / "output.md"
    json_path = run_dir / "output.json"

    markdown = call_first(doc, ["export_to_markdown", "export_to_md"], default="")
    md_path.write_text(str(markdown or ""), encoding="utf-8")

    json_value = call_first(doc, ["export_to_dict", "export_to_json"], default={})
    if isinstance(json_value, str):
        try:
            parsed = json.loads(json_value)
        except Exception:
            parsed = {"raw_json_export": json_value}
    else:
        parsed = json_value
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(make_jsonable(parsed), handle, ensure_ascii=False, indent=2)

    return {"markdown": str(md_path), "json": str(json_path)}


def call_first(obj: Any, names: list[str], default: Any = None) -> Any:
    for name in names:
        method = getattr(obj, name, None)
        if callable(method):
            return method()
    return default


def make_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        if isinstance(value, dict):
            return {str(k): make_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [make_jsonable(item) for item in value]
        return str(value)


def extract_tables(doc: Any, tables_dir: Path) -> int:
    tables_dir.mkdir(parents=True, exist_ok=True)
    tables = getattr(doc, "tables", []) or []
    count = 0
    for index, table in enumerate(tables, start=1):
        stem = tables_dir / f"table_{index:03d}"
        try:
            dataframe = table_to_dataframe(table)
            if dataframe is not None:
                dataframe.to_csv(stem.with_suffix(".csv"), index=False)
                dataframe.to_html(stem.with_suffix(".html"), index=False)
                count += 1
                continue

            html = table_to_html(table)
            if html:
                stem.with_suffix(".html").write_text(html, encoding="utf-8")
                count += 1
        except Exception:
            LOGGER.exception("Failed to export table %s", index)
    return count


def table_to_dataframe(table: Any) -> Any | None:
    import pandas as pd

    for name in ["export_to_dataframe", "to_dataframe"]:
        method = getattr(table, name, None)
        if callable(method):
            try:
                return method()
            except TypeError:
                continue

    data = getattr(table, "data", None)
    if data is not None:
        for name in ["export_to_dataframe", "to_dataframe"]:
            method = getattr(data, name, None)
            if callable(method):
                try:
                    return method()
                except TypeError:
                    continue
        if isinstance(data, list):
            return pd.DataFrame(data)

    return None


def table_to_html(table: Any) -> str | None:
    for target in [table, getattr(table, "data", None)]:
        if target is None:
            continue
        for name in ["export_to_html", "to_html"]:
            method = getattr(target, name, None)
            if callable(method):
                try:
                    return str(method())
                except TypeError:
                    continue
    return None


def save_images(doc: Any, images_dir: Path) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for collection_name, prefix in [("pages", "page"), ("pictures", "picture")]:
        collection = getattr(doc, collection_name, None)
        items = collection.values() if isinstance(collection, dict) else (collection or [])
        for index, item in enumerate(items, start=1):
            image = find_image_object(item)
            if image is None:
                continue
            path = images_dir / f"{prefix}_{index:03d}.png"
            try:
                image.save(path)
                count += 1
            except Exception:
                LOGGER.debug("Could not save image for %s %s", collection_name, index, exc_info=True)
    return count


def find_image_object(item: Any) -> Any | None:
    candidates = [
        item,
        getattr(item, "image", None),
        getattr(getattr(item, "image", None), "pil_image", None),
        getattr(item, "pil_image", None),
    ]
    for candidate in candidates:
        if hasattr(candidate, "save") and callable(candidate.save):
            return candidate
    return None


def maybe_page_count(result: Any, doc: Any) -> int | None:
    for obj in [result, doc]:
        for attr in ["page_count", "num_pages"]:
            value = getattr(obj, attr, None)
            if isinstance(value, int):
                return value
    pages = getattr(doc, "pages", None)
    if pages is not None:
        try:
            return len(pages)
        except Exception:
            return None
    return None


def profile_should_skip(
    profile_name: str,
    profile: dict[str, Any],
    args: argparse.Namespace,
    gpu_info: dict[str, Any],
    explicitly_selected: bool,
) -> str | None:
    gpu_required = bool(profile.get("gpu_required", False))
    selected_device = args.device_override or str(profile.get("device", "auto"))
    cuda_requested = selected_device == "cuda"

    if gpu_required and not args.allow_gpu:
        return "GPU-required profile skipped because --allow-gpu was not passed."

    if (gpu_required or cuda_requested) and not gpu_info["available"]:
        reason = "Profile requires CUDA/GPU, but no NVIDIA GPU was detected."
        if args.allow_missing_gpu:
            return reason
        if explicitly_selected or gpu_required:
            raise RuntimeError(
                f"{profile_name}: {reason} Re-run on a GPU host or pass --allow-missing-gpu to record a skip."
            )
    return None


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(make_jsonable(metadata), handle, ensure_ascii=False, indent=2)


def write_profile(path: Path, profile: dict[str, Any]) -> None:
    import yaml

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(profile, handle, sort_keys=False)


def run_one(
    pdf_path: Path,
    profile_name: str,
    profile: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
    api: dict[str, Any] | None,
    gpu_before: dict[str, Any],
) -> dict[str, Any]:
    run_dir = output_root / pdf_path.stem / profile_name
    tables_dir = run_dir / "tables"
    images_dir = run_dir / "images"
    metadata_path = run_dir / "run_metadata.json"

    selected_device = args.device_override or str(profile.get("device", "auto"))
    row: dict[str, Any] = {
        "pdf_filename": pdf_path.name,
        "profile": profile_name,
        "status": "pending",
        "runtime_seconds": None,
        "selected_device": selected_device,
        "gpu_available": gpu_before["available"],
        "gpu_name": gpu_before["name"],
        "number_of_tables": None,
        "markdown_output_path": None,
        "json_output_path": None,
        "error_or_skip_reason": None,
    }

    if run_dir.exists() and not args.overwrite and (run_dir / "run_metadata.json").exists():
        row["status"] = "skipped"
        row["error_or_skip_reason"] = "Existing output found; pass --overwrite to rerun."
        return row

    run_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    start = utc_now()
    start_time = time.perf_counter()
    metadata = {
        "pdf_path": str(pdf_path),
        "pdf_filename": pdf_path.name,
        "profile_name": profile_name,
        "profile_description": profile.get("description"),
        "docling_version": docling_version(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "start_timestamp": start,
        "end_timestamp": None,
        "runtime_seconds": None,
        "status": "failure",
        "error_message": None,
        "gpu_available": gpu_before["available"],
        "gpu_name": gpu_before["name"],
        "gpu_memory_before": gpu_before,
        "gpu_memory_after": None,
        "selected_device": selected_device,
        "page_count": None,
        "number_of_tables_extracted": None,
        "output_file_paths": {},
    }

    write_profile(run_dir / "profile_used.yaml", {**profile, "selected_device": selected_device})
    error_log = run_dir / "errors.log"
    error_log.write_text("", encoding="utf-8")

    try:
        if api is None:
            api = import_docling_api()
        pipeline_options = build_pipeline_options(profile, selected_device, api)
        converter = build_converter(profile, pipeline_options, api)
        result = convert_pdf(converter, pdf_path, args.max_pages)
        doc = getattr(result, "document", result)

        output_paths = export_document(doc, run_dir)
        table_count = extract_tables(doc, tables_dir)
        image_count = save_images(doc, images_dir)

        metadata["page_count"] = maybe_page_count(result, doc)
        metadata["number_of_tables_extracted"] = table_count
        metadata["output_file_paths"] = {
            **output_paths,
            "profile_used": str(run_dir / "profile_used.yaml"),
            "metadata": str(metadata_path),
            "tables_dir": str(tables_dir),
            "images_dir": str(images_dir),
            "images_saved": image_count,
        }
        metadata["status"] = "success"
        row["status"] = "success"
        row["number_of_tables"] = table_count
        row["markdown_output_path"] = output_paths.get("markdown")
        row["json_output_path"] = output_paths.get("json")
    except Exception as exc:
        message = str(exc)
        metadata["error_message"] = message
        metadata["status"] = "failure"
        row["status"] = "failure"
        row["error_or_skip_reason"] = message
        error_log.write_text(traceback.format_exc(), encoding="utf-8")
        LOGGER.error("%s / %s failed: %s", pdf_path.name, profile_name, message)
    finally:
        runtime = round(time.perf_counter() - start_time, 3)
        gpu_after = detect_gpu()
        metadata["end_timestamp"] = utc_now()
        metadata["runtime_seconds"] = runtime
        metadata["gpu_memory_after"] = gpu_after
        row["runtime_seconds"] = runtime
        row["gpu_available"] = gpu_after["available"]
        row["gpu_name"] = gpu_after["name"] or row["gpu_name"]
        write_metadata(metadata_path, metadata)

    return row


def skipped_row(
    pdf_path: Path,
    profile_name: str,
    profile: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
    gpu_info: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    run_dir = output_root / pdf_path.stem / profile_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(exist_ok=True)
    (run_dir / "images").mkdir(exist_ok=True)
    (run_dir / "errors.log").write_text(reason + "\n", encoding="utf-8")
    selected_device = args.device_override or str(profile.get("device", "auto"))
    write_profile(run_dir / "profile_used.yaml", {**profile, "selected_device": selected_device})
    metadata = {
        "pdf_path": str(pdf_path),
        "pdf_filename": pdf_path.name,
        "profile_name": profile_name,
        "profile_description": profile.get("description"),
        "docling_version": docling_version(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "start_timestamp": utc_now(),
        "end_timestamp": utc_now(),
        "runtime_seconds": 0,
        "status": "skipped",
        "error_message": reason,
        "gpu_available": gpu_info["available"],
        "gpu_name": gpu_info["name"],
        "gpu_memory_before": gpu_info,
        "gpu_memory_after": gpu_info,
        "selected_device": selected_device,
        "page_count": None,
        "number_of_tables_extracted": None,
        "output_file_paths": {
            "profile_used": str(run_dir / "profile_used.yaml"),
            "metadata": str(run_dir / "run_metadata.json"),
            "tables_dir": str(run_dir / "tables"),
            "images_dir": str(run_dir / "images"),
        },
    }
    write_metadata(run_dir / "run_metadata.json", metadata)
    return {
        "pdf_filename": pdf_path.name,
        "profile": profile_name,
        "status": "skipped",
        "runtime_seconds": 0,
        "selected_device": selected_device,
        "gpu_available": gpu_info["available"],
        "gpu_name": gpu_info["name"],
        "number_of_tables": None,
        "markdown_output_path": None,
        "json_output_path": None,
        "error_or_skip_reason": reason,
    }


def write_summary(rows: list[dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    columns = [
        "pdf_filename",
        "profile",
        "status",
        "runtime_seconds",
        "selected_device",
        "gpu_available",
        "gpu_name",
        "number_of_tables",
        "markdown_output_path",
        "json_output_path",
        "error_or_skip_reason",
    ]
    try:
        import pandas as pd

        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(output_root / "summary.csv", index=False)
        df.to_excel(output_root / "summary.xlsx", index=False)
        try:
            markdown = df.to_markdown(index=False)
        except Exception:
            markdown = dataframe_to_markdown(rows, columns)
        (output_root / "summary.md").write_text(markdown + "\n", encoding="utf-8")
    except Exception as exc:
        LOGGER.warning("Falling back to standard-library summary writer: %s", exc)
        write_csv_summary(rows, columns, output_root / "summary.csv")
        (output_root / "summary.md").write_text(dataframe_to_markdown(rows, columns) + "\n", encoding="utf-8")
        write_minimal_xlsx(rows, columns, output_root / "summary.xlsx")


def write_csv_summary(rows: list[dict[str, Any]], columns: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def dataframe_to_markdown(rows: list[dict[str, Any]], columns: list[str]) -> str:
    values = [[stringify_cell(row.get(column)) for column in columns] for row in rows]
    widths = [
        max([len(column)] + [len(row[index]) for row in values])
        for index, column in enumerate(columns)
    ]
    header = "| " + " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[index] for index in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(columns))) + " |"
        for row in values
    ]
    return "\n".join([header, separator, *body])


def stringify_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ")


def write_minimal_xlsx(rows: list[dict[str, Any]], columns: list[str], path: Path) -> None:
    sheet_rows = [columns] + [[stringify_cell(row.get(column)) for column in columns] for row in rows]
    sheet_xml = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "<sheetData>",
    ]
    for row_index, row in enumerate(sheet_rows, start=1):
        sheet_xml.append(f'<row r="{row_index}">')
        for col_index, value in enumerate(row, start=1):
            cell_ref = f"{xlsx_column_name(col_index)}{row_index}"
            sheet_xml.append(
                f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
            )
        sheet_xml.append("</row>")
    sheet_xml.extend(["</sheetData>", "</worksheet>"])

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>""",
        )
        archive.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
        )
        archive.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="summary" sheetId="1" r:id="rId1"/></sheets>
</workbook>""",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>""",
        )
        archive.writestr("xl/worksheets/sheet1.xml", "\n".join(sheet_xml))


def xlsx_column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def main() -> int:
    setup_logging()
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    profiles = load_profiles(config_path)
    selected_profile_names = select_profiles(profiles, args.profiles)
    explicitly_selected = args.profiles.strip().lower() != "all"
    pdfs = find_pdfs(input_path)
    if not pdfs:
        LOGGER.warning("No PDFs found under %s", input_path)

    gpu_info = detect_gpu()
    LOGGER.info(
        "GPU available: %s%s",
        gpu_info["available"],
        f" ({gpu_info['name']})" if gpu_info.get("name") else "",
    )

    api: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    total = len(pdfs) * len(selected_profile_names)

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = SimpleProgress  # type: ignore[assignment]

    with tqdm(total=total, desc="Docling ablation", unit="run") as progress:
        for pdf_path in pdfs:
            for profile_name in selected_profile_names:
                profile = dict(profiles[profile_name])
                selected_device = args.device_override or str(profile.get("device", "auto"))
                profile["device"] = selected_device
                try:
                    skip_reason = profile_should_skip(
                        profile_name, profile, args, gpu_info, explicitly_selected
                    )
                    if skip_reason:
                        rows.append(
                            skipped_row(pdf_path, profile_name, profile, args, output_root, gpu_info, skip_reason)
                        )
                    else:
                        rows.append(run_one(pdf_path, profile_name, profile, args, output_root, api, gpu_info))
                except Exception as exc:
                    if rows:
                        write_summary(rows, output_root)
                    LOGGER.error(str(exc))
                    raise
                finally:
                    progress.update(1)

    write_summary(rows, output_root)
    LOGGER.info("Wrote summary files to %s", output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
