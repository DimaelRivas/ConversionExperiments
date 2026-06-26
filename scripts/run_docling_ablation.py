#!/usr/bin/env python3
"""Run Docling extraction ablations over one PDF or a PDF folder."""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import logging
import os
import pkgutil
import platform
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

from docling_gpu_utils import detect_gpu
from docling_profile_utils import (
    normalize_profile,
    normalize_profiles,
    output_exists,
    plan_skip_reason,
    profile_table_rows,
    select_profile_names,
    selected_device,
)
from docling_table_metrics import (
    SCORING_CONFIG,
    TABLE_METRIC_KEYS,
    build_inspection_markdown,
    compute_table_metrics,
    empty_metrics,
    export_tables,
    score_candidate,
    write_table_metrics,
)


LOGGER = logging.getLogger("docling_ablation")

SUMMARY_COLUMNS = [
    "pdf_filename",
    "profile",
    "profile_group",
    "quality_tier",
    "resource_class",
    "status",
    "skip_reason",
    "error_message",
    "runtime_seconds",
    "pipeline",
    "pdf_backend",
    "table_mode",
    "do_cell_matching",
    "ocr_engine",
    "ocr_lang",
    "psm",
    "do_ocr",
    "force_full_page_ocr",
    "vlm_model",
    "images_scale",
    "generate_page_images",
    "generate_picture_images",
    "device_requested",
    "device_selected",
    "gpu_available",
    "gpu_name",
    "gpu_memory_total_mb",
    "gpu_memory_before_mb",
    "gpu_memory_after_mb",
    "page_count",
    "table_count",
    "markdown_output_path",
    "json_output_path",
    "run_dir",
]

TABLE_SUMMARY_COLUMNS = [
    "pdf_filename",
    "profile",
    "profile_group",
    "quality_tier",
    "resource_class",
    "status",
    "runtime_seconds",
    "pipeline",
    "pdf_backend",
    "ocr_engine",
    "psm",
    "do_cell_matching",
    *TABLE_METRIC_KEYS,
    "table_metrics_path",
    "inspection_path",
    "run_dir",
]

BEST_CANDIDATE_COLUMNS = [
    "pdf_filename",
    "profile",
    "heuristic_score",
    "profile_group",
    "quality_tier",
    "resource_class",
    "status",
    "runtime_seconds",
    "pipeline",
    "pdf_backend",
    "ocr_engine",
    "psm",
    "do_cell_matching",
    "table_count",
    "tables_with_rows",
    "total_rows",
    "total_columns_sum",
    "max_columns",
    "max_rows",
    "empty_cell_ratio",
    "single_column_table_count",
    "very_small_table_count",
    "inspection_path",
    "run_dir",
]


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


class UnsupportedFeature(RuntimeError):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass
class BuildContext:
    warnings: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Docling PDF extraction ablation profiles and save structured outputs."
    )
    parser.add_argument("--input", help="PDF file or folder containing PDFs.")
    parser.add_argument("--profiles", default="all", help="'all' or comma-separated profile names.")
    parser.add_argument("--config", default="configs/docling_ablation_profiles.yaml")
    parser.add_argument("--output", default="outputs/docling_ablation")
    parser.add_argument("--allow-gpu", action="store_true", help="Enable GPU-required/CUDA profiles.")
    parser.add_argument(
        "--allow-missing-gpu",
        action="store_true",
        help="Skip selected GPU-required profiles when no GPU is available instead of recording a failure.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rerun even if outputs already exist.")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional page limit for quick tests.")
    parser.add_argument(
        "--device-override",
        choices=["cpu", "auto", "cuda"],
        default=None,
        help="Override profile device selection.",
    )
    parser.add_argument(
        "--profile-groups",
        default=None,
        help="Comma-separated group filter, e.g. native_text,ocr,backend,vlm,heavy_quality.",
    )
    parser.add_argument(
        "--quality-tiers",
        default=None,
        help="Comma-separated quality tier filter, e.g. strong,heavy,extreme.",
    )
    parser.add_argument(
        "--resource-classes",
        default=None,
        help="Comma-separated resource class filter, e.g. high,extreme.",
    )
    parser.add_argument("--list-profiles", action="store_true", help="List selected profiles and exit.")
    parser.add_argument("--dry-run-plan", action="store_true", help="Print planned PDF/profile runs and exit.")
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
    return normalize_profiles(profiles)


def find_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {input_path}")
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*.pdf") if path.is_file())
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def docling_version() -> str:
    try:
        import docling  # type: ignore

        value = getattr(docling, "__version__", None)
        if value:
            return str(value)
    except Exception:
        pass
    try:
        from importlib.metadata import version

        return version("docling")
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
        "backend_classes": discover_pdf_backends(),
    }

    for name in [
        "EasyOcrOptions",
        "TesseractCliOcrOptions",
        "TesseractOcrOptions",
        "RapidOcrOptions",
        "RapidOCROptions",
        "AcceleratorOptions",
        "AcceleratorDevice",
        "VlmPipelineOptions",
        "GraniteDoclingVlmOptions",
        "SmolDoclingVlmOptions",
        "SmolDoclingVlmOptions",
        "QwenVlmOptions",
        "Qwen25VlmOptions",
        "_default_vlm_convert_options",
        "smoldocling_vlm_conversion_options",
        "granite_docling_vlm_conversion_options",
        "granite_vision_vlm_conversion_options",
        "qwen_vlm_conversion_options",
        "qwen2_vlm_conversion_options",
        "qwen25_vlm_conversion_options",
    ]:
        try:
            module = __import__("docling.datamodel.pipeline_options", fromlist=[name])
            api[name] = getattr(module, name)
        except Exception:
            api[name] = None

    for pipeline_name, import_path, attr_name in [
        ("StandardPdfPipeline", "docling.pipeline.standard_pdf_pipeline", "StandardPdfPipeline"),
        ("VlmPipeline", "docling.pipeline.vlm_pipeline", "VlmPipeline"),
    ]:
        try:
            module = __import__(import_path, fromlist=[attr_name])
            api[pipeline_name] = getattr(module, attr_name)
        except Exception:
            api[pipeline_name] = None

    return api


def discover_pdf_backends() -> dict[str, Any]:
    backends: dict[str, Any] = {}

    candidates = {
        "docling_parse": [
            ("docling.backend.docling_parse_backend", "DoclingParseDocumentBackend"),
            ("docling.backend.docling_parse_v2_backend", "DoclingParseV2DocumentBackend"),
            ("docling.backend.docling_parse_v4_backend", "DoclingParseV4DocumentBackend"),
        ],
        "pypdfium2": [
            ("docling.backend.pypdfium2_backend", "PyPdfiumDocumentBackend"),
            ("docling.backend.pypdfium2_backend", "PyPdfium2DocumentBackend"),
        ],
        "threaded_docling_parse": [
            ("docling.backend.docling_parse_backend", "ThreadedDoclingParseDocumentBackend"),
            ("docling.backend.docling_parse_v2_backend", "ThreadedDoclingParseV2DocumentBackend"),
            ("docling.backend.docling_parse_v4_backend", "ThreadedDoclingParseV4DocumentBackend"),
        ],
    }

    for logical_name, class_candidates in candidates.items():
        for module_name, class_name in class_candidates:
            backend_cls = import_attr(module_name, class_name)
            if backend_cls is not None:
                backends[logical_name] = backend_cls
                break

    try:
        import docling.backend as backend_pkg  # type: ignore

        for module_info in pkgutil.iter_modules(backend_pkg.__path__, backend_pkg.__name__ + "."):
            if not module_info.name.endswith("_backend"):
                continue
            try:
                module = __import__(module_info.name, fromlist=["*"])
            except Exception:
                continue
            for attr_name in dir(module):
                if not attr_name.endswith("DocumentBackend"):
                    continue
                backend_cls = getattr(module, attr_name, None)
                if inspect.isclass(backend_cls):
                    key = attr_name.replace("DocumentBackend", "")
                    key = camel_to_snake(key).replace("_document", "").strip("_")
                    backends.setdefault(key, backend_cls)
    except Exception:
        pass

    return backends


def import_attr(module_name: str, attr_name: str) -> Any | None:
    try:
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    except Exception:
        return None


def camel_to_snake(value: str) -> str:
    chars: list[str] = []
    for index, char in enumerate(value):
        if char.isupper() and index > 0 and value[index - 1].islower():
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


def build_pipeline_options(
    profile: dict[str, Any],
    requested_device: str,
    api: dict[str, Any],
    context: BuildContext,
) -> Any:
    pipeline_type = str(profile.get("pipeline", "standard")).lower()
    options_cls = api["PdfPipelineOptions"]
    if pipeline_type == "vlm":
        options_cls = api.get("VlmPipelineOptions")
        if options_cls is None:
            raise UnsupportedFeature(
                "unsupported_vlm_model_or_docling_api",
                "This Docling version does not expose VlmPipelineOptions.",
            )

    options = options_cls()
    apply_optional(
        options,
        "do_ocr",
        bool(profile.get("do_ocr", False)),
        context,
        warn_if_unsupported=bool(profile.get("do_ocr", False)),
    )
    apply_optional(
        options,
        "force_full_page_ocr",
        bool(profile.get("force_full_page_ocr", False)),
        context,
        warn_if_unsupported=bool(profile.get("force_full_page_ocr", False)),
    )
    apply_optional(options, "do_table_structure", bool(profile.get("do_table_structure", True)), context)
    apply_optional(options, "generate_page_images", bool(profile.get("generate_page_images", False)), context)
    apply_optional(options, "generate_picture_images", bool(profile.get("generate_picture_images", False)), context)
    apply_optional(options, "images_scale", float(profile.get("images_scale", 1.0)), context)

    for attr in ["page_batch_size", "layout_batch_size"]:
        if profile.get(attr) is not None:
            apply_optional(options, attr, profile.get(attr), context)

    table_options = getattr(options, "table_structure_options", None)
    if table_options is not None:
        apply_optional(table_options, "do_cell_matching", bool(profile.get("do_cell_matching", True)), context)
        mode = table_mode(profile.get("table_mode", "accurate"), api["TableFormerMode"])
        apply_optional(table_options, "mode", mode, context)
        if profile.get("table_batch_size") is not None:
            apply_optional(table_options, "table_batch_size", profile.get("table_batch_size"), context)
    elif profile.get("do_table_structure", True):
        context.warnings.append("table_structure_options_unavailable")

    if bool(profile.get("do_ocr", False)):
        ocr_options = build_ocr_options(profile, requested_device, api, context)
        if not apply_optional(options, "ocr_options", ocr_options, context):
            raise UnsupportedFeature(
                "unsupported_ocr_engine_or_docling_api",
                "Pipeline options do not expose ocr_options in this Docling version.",
            )

    accelerator = build_accelerator_options(requested_device, profile, api, context)
    if accelerator is not None:
        apply_optional(options, "accelerator_options", accelerator, context)

    if pipeline_type == "vlm":
        configure_vlm_options(options, profile, api, context)

    return options


def apply_optional(
    obj: Any,
    attr: str,
    value: Any,
    context: BuildContext,
    warn_if_unsupported: bool = True,
) -> bool:
    if hasattr(obj, attr):
        try:
            setattr(obj, attr, value)
            return True
        except Exception as exc:
            context.warnings.append(f"{attr}_not_set:{exc}")
            return False
    if warn_if_unsupported:
        context.warnings.append(f"{attr}_unsupported_by_docling_api")
    return False


def table_mode(mode_name: str, enum_cls: Any) -> Any:
    normalized = str(mode_name).strip().lower()
    if normalized == "fast":
        return getattr(enum_cls, "FAST")
    if normalized == "accurate":
        return getattr(enum_cls, "ACCURATE")
    raise ValueError(f"Unsupported table_mode: {mode_name}")


def build_ocr_options(
    profile: dict[str, Any],
    requested_device: str,
    api: dict[str, Any],
    context: BuildContext,
) -> Any:
    engine = str(profile.get("ocr_engine", "easyocr")).lower()
    languages = list(profile.get("ocr_lang") or [])
    psm = profile.get("psm")

    if engine == "tesseract":
        option_cls = api.get("TesseractCliOcrOptions") or api.get("TesseractOcrOptions")
        if option_cls is None:
            raise UnsupportedFeature(
                "unsupported_ocr_engine_or_docling_api",
                "This Docling version does not expose Tesseract OCR options.",
            )
        option = instantiate_compatible(option_cls, {"lang": languages, "languages": languages, "psm": psm})
        if psm is not None and not has_or_accepts(option_cls, option, "psm"):
            raise UnsupportedFeature(
                "unsupported_ocr_engine_or_docling_api",
                "Tesseract OCR options do not expose psm in this Docling version.",
            )
        if languages and not (has_or_accepts(option_cls, option, "lang") or has_or_accepts(option_cls, option, "languages")):
            context.warnings.append("tesseract_language_option_unsupported")
        set_remaining_fields(option, {"lang": languages, "languages": languages, "psm": psm}, context)
        return option

    if engine == "easyocr":
        option_cls = api.get("EasyOcrOptions")
        if option_cls is None:
            raise UnsupportedFeature(
                "unsupported_ocr_engine_or_docling_api",
                "This Docling version does not expose EasyOcrOptions.",
            )
        option = instantiate_compatible(option_cls, {"lang": languages, "languages": languages})
        set_remaining_fields(option, {"lang": languages, "languages": languages}, context)
        if requested_device == "cuda":
            for attr in ["use_gpu", "gpu"]:
                if hasattr(option, attr):
                    setattr(option, attr, True)
        return option

    if engine == "rapidocr":
        option_cls = api.get("RapidOcrOptions") or api.get("RapidOCROptions")
        if option_cls is None:
            raise UnsupportedFeature(
                "unsupported_ocr_engine_or_docling_api",
                "This Docling version does not expose RapidOCR options.",
            )
        backend_requested = profile.get("rapidocr_backend")
        values = {
            "lang": languages,
            "languages": languages,
            "backend": backend_requested,
            "rapidocr_backend": backend_requested,
        }
        option = instantiate_compatible(option_cls, values)
        set_remaining_fields(option, values, context)
        if backend_requested and not (
            has_or_accepts(option_cls, option, "backend")
            or has_or_accepts(option_cls, option, "rapidocr_backend")
        ):
            context.warnings.append("rapidocr_backend_unsupported_by_docling_api")
        return option

    raise UnsupportedFeature("unsupported_ocr_engine_or_docling_api", f"Unsupported OCR engine: {engine}")


def instantiate_compatible(cls: Any, values: dict[str, Any]) -> Any:
    kwargs = compatible_kwargs(cls, {key: value for key, value in values.items() if value is not None})
    try:
        return cls(**kwargs)
    except Exception:
        return cls()


def compatible_kwargs(cls: Any, values: dict[str, Any]) -> dict[str, Any]:
    try:
        params = inspect.signature(cls).parameters
        return {key: value for key, value in values.items() if key in params}
    except Exception:
        return values


def has_or_accepts(cls: Any, obj: Any, attr: str) -> bool:
    if hasattr(obj, attr):
        return True
    try:
        return attr in inspect.signature(cls).parameters
    except Exception:
        return False


def set_remaining_fields(obj: Any, values: dict[str, Any], context: BuildContext) -> None:
    for attr, value in values.items():
        if value is None:
            continue
        if hasattr(obj, attr):
            try:
                setattr(obj, attr, value)
            except Exception as exc:
                context.warnings.append(f"{attr}_not_set:{exc}")


def build_accelerator_options(
    requested_device: str,
    profile: dict[str, Any],
    api: dict[str, Any],
    context: BuildContext,
) -> Any | None:
    accelerator_cls = api.get("AcceleratorOptions")
    if accelerator_cls is None:
        context.warnings.append("accelerator_options_unsupported_by_docling_api")
        return None

    accelerator_device = api.get("AcceleratorDevice")
    value: Any = requested_device
    if accelerator_device is not None:
        lookup = {"cpu": "CPU", "cuda": "CUDA", "auto": "AUTO"}
        enum_name = lookup.get(requested_device, "AUTO")
        value = getattr(accelerator_device, enum_name, requested_device)

    kwargs: dict[str, Any] = {"device": value}
    if profile.get("num_threads") is not None:
        kwargs["num_threads"] = profile.get("num_threads")

    try:
        return accelerator_cls(**compatible_kwargs(accelerator_cls, kwargs))
    except Exception:
        obj = accelerator_cls()
        apply_optional(obj, "device", value, context)
        if profile.get("num_threads") is not None:
            apply_optional(obj, "num_threads", profile.get("num_threads"), context)
        return obj


def configure_vlm_options(options: Any, profile: dict[str, Any], api: dict[str, Any], context: BuildContext) -> None:
    model_name = str(profile.get("vlm_model", "")).lower()
    if model_name == "granite_docling":
        granite_cls = api.get("GraniteDoclingVlmOptions")
        if granite_cls is not None:
            try:
                vlm_options = granite_cls()
            except Exception as exc:
                raise UnsupportedFeature(
                    "unsupported_vlm_model_or_docling_api",
                    f"Could not instantiate Granite Docling VLM options: {exc}",
                ) from exc
            set_vlm_runtime_options(vlm_options, profile, context)
            if not apply_optional(options, "vlm_options", vlm_options, context):
                raise UnsupportedFeature(
                    "unsupported_vlm_model_or_docling_api",
                    "VLM pipeline options do not expose vlm_options.",
                )
            return

        current = getattr(options, "vlm_options", None)
        current_name = str(getattr(getattr(current, "model_spec", None), "name", "")).lower()
        if current is not None and "granite" in current_name and "docling" in current_name:
            set_vlm_runtime_options(current, profile, context)
            return

        preset = api.get("granite_docling_vlm_conversion_options") or api.get("_default_vlm_convert_options")
        if preset is not None:
            vlm_options = copy.deepcopy(preset)
            set_vlm_runtime_options(vlm_options, profile, context)
            if not apply_optional(options, "vlm_options", vlm_options, context):
                raise UnsupportedFeature(
                    "unsupported_vlm_model_or_docling_api",
                    "VLM pipeline options do not expose vlm_options.",
                )
            return

        raise UnsupportedFeature(
            "unsupported_vlm_model_or_docling_api",
            "This Docling version does not expose Granite Docling VLM options.",
        )

    preset_names = {
        "smoldocling": ["smoldocling_vlm_conversion_options"],
        "qwen": ["qwen_vlm_conversion_options", "qwen2_vlm_conversion_options", "qwen25_vlm_conversion_options"],
        "qwen_cuda_if_available": [
            "qwen_vlm_conversion_options",
            "qwen2_vlm_conversion_options",
            "qwen25_vlm_conversion_options",
        ],
    }
    preset = next((api.get(name) for name in preset_names.get(model_name, []) if api.get(name) is not None), None)
    if preset is not None:
        vlm_options = copy.deepcopy(preset)
        set_vlm_runtime_options(vlm_options, profile, context)
        if not apply_optional(options, "vlm_options", vlm_options, context):
            raise UnsupportedFeature(
                "unsupported_vlm_model_or_docling_api",
                "VLM pipeline options do not expose vlm_options.",
            )
        return

    model_class_names = {
        "smoldocling": ["SmolDoclingVlmOptions"],
        "qwen": ["QwenVlmOptions", "Qwen25VlmOptions"],
        "qwen_cuda_if_available": ["QwenVlmOptions", "Qwen25VlmOptions"],
    }

    class_names = model_class_names.get(model_name)
    if not class_names:
        raise UnsupportedFeature(
            "unsupported_vlm_model_or_docling_api",
            f"No VLM option mapping is defined for {model_name}.",
        )

    option_cls = next((api.get(name) for name in class_names if api.get(name) is not None), None)
    if option_cls is None:
        raise UnsupportedFeature(
            "unsupported_vlm_model_or_docling_api",
            f"This Docling version does not expose VLM options for {model_name}.",
        )

    try:
        vlm_options = option_cls()
    except Exception as exc:
        raise UnsupportedFeature(
            "unsupported_vlm_model_or_docling_api",
            f"Could not instantiate VLM options for {model_name}: {exc}",
        ) from exc

    set_vlm_runtime_options(vlm_options, profile, context)
    if not apply_optional(options, "vlm_options", vlm_options, context):
        raise UnsupportedFeature(
            "unsupported_vlm_model_or_docling_api",
            "VLM pipeline options do not expose vlm_options.",
        )


def set_vlm_runtime_options(vlm_options: Any, profile: dict[str, Any], context: BuildContext) -> None:
    if profile.get("images_scale") is not None:
        apply_optional(vlm_options, "scale", float(profile.get("images_scale")), context, warn_if_unsupported=False)
    if profile.get("page_batch_size") is not None:
        apply_optional(vlm_options, "batch_size", profile.get("page_batch_size"), context, warn_if_unsupported=False)


def build_converter(profile: dict[str, Any], pipeline_options: Any, api: dict[str, Any]) -> Any:
    DocumentConverter = api["DocumentConverter"]
    PdfFormatOption = api["PdfFormatOption"]
    InputFormat = api["InputFormat"]

    kwargs: dict[str, Any] = {"pipeline_options": pipeline_options}
    if str(profile.get("pipeline", "standard")).lower() == "vlm":
        if api.get("VlmPipeline") is None:
            raise UnsupportedFeature(
                "unsupported_vlm_model_or_docling_api",
                "This Docling version does not expose VlmPipeline.",
            )
        kwargs["pipeline_cls"] = api["VlmPipeline"]
    elif api.get("StandardPdfPipeline") is not None:
        kwargs["pipeline_cls"] = api["StandardPdfPipeline"]

    backend_name = profile.get("pdf_backend")
    if backend_name:
        backend_cls = api.get("backend_classes", {}).get(str(backend_name))
        if backend_cls is None:
            raise UnsupportedFeature(
                "unsupported_pdf_backend_or_docling_api",
                f"PDF backend is not available in this Docling installation: {backend_name}",
            )
        kwargs["backend"] = backend_cls
        kwargs["backend_cls"] = backend_cls

    pdf_option = instantiate_format_option(PdfFormatOption, kwargs, require_backend=bool(backend_name))
    return DocumentConverter(format_options={InputFormat.PDF: pdf_option})


def instantiate_format_option(cls: Any, kwargs: dict[str, Any], require_backend: bool = False) -> Any:
    candidate_kwargs: list[dict[str, Any]] = []
    if require_backend:
        for backend_key in ["backend", "backend_cls"]:
            if backend_key in kwargs:
                candidate = dict(kwargs)
                other = "backend_cls" if backend_key == "backend" else "backend"
                candidate.pop(other, None)
                candidate_kwargs.append(candidate)
    else:
        candidate_kwargs.append(dict(kwargs))

    for candidate in candidate_kwargs:
        compatible = compatible_kwargs(cls, candidate)
        if require_backend and "backend" not in compatible and "backend_cls" not in compatible:
            continue
        try:
            return cls(**compatible)
        except Exception:
            without_pipeline = dict(compatible)
            without_pipeline.pop("pipeline_cls", None)
            try:
                return cls(**without_pipeline)
            except Exception:
                continue

    if require_backend:
        raise UnsupportedFeature(
            "unsupported_pdf_backend_or_docling_api",
            "PdfFormatOption does not accept an explicit PDF backend in this Docling version.",
        )

    compatible = compatible_kwargs(cls, kwargs)
    compatible.pop("backend_cls", None)
    compatible.pop("backend", None)
    compatible.pop("pipeline_cls", None)
    return cls(**compatible)


def convert_pdf(converter: Any, pdf_path: Path, max_pages: int | None) -> tuple[Any, list[str]]:
    warnings: list[str] = []
    if max_pages is None:
        return converter.convert(pdf_path), warnings

    try:
        params = inspect.signature(converter.convert).parameters
    except Exception:
        params = {}

    if "page_range" in params:
        return converter.convert(pdf_path, page_range=(1, max_pages)), warnings
    if "page_ranges" in params:
        return converter.convert(pdf_path, page_ranges=[(1, max_pages)]), warnings
    if "pages" in params:
        return converter.convert(pdf_path, pages=range(1, max_pages + 1)), warnings

    warnings.append("max_pages_unsupported_by_docling_api")
    return converter.convert(pdf_path), warnings


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


def base_metadata(
    pdf_path: Path,
    profile_name: str,
    profile: dict[str, Any],
    device: str,
    gpu_info: dict[str, Any],
    status: str = "failure",
) -> dict[str, Any]:
    return {
        "pdf_path": str(pdf_path),
        "pdf_filename": pdf_path.name,
        "profile_name": profile_name,
        "profile_description": profile.get("description"),
        "profile_group": profile.get("group"),
        "quality_tier": profile.get("quality_tier"),
        "resource_class": profile.get("resource_class"),
        "pdf_backend": profile.get("pdf_backend"),
        "table_mode": profile.get("table_mode"),
        "do_cell_matching": profile.get("do_cell_matching"),
        "ocr_engine": profile.get("ocr_engine"),
        "ocr_lang": profile.get("ocr_lang"),
        "psm": profile.get("psm"),
        "do_ocr": profile.get("do_ocr"),
        "force_full_page_ocr": profile.get("force_full_page_ocr"),
        "pipeline": profile.get("pipeline"),
        "vlm_model": profile.get("vlm_model"),
        "images_scale": profile.get("images_scale"),
        "generate_page_images": profile.get("generate_page_images"),
        "generate_picture_images": profile.get("generate_picture_images"),
        "device_requested": device,
        "device_selected": device,
        "gpu_available": gpu_info.get("available"),
        "gpu_name": gpu_info.get("name"),
        "gpu_memory_total_mb": gpu_info.get("memory_total_mb"),
        "gpu_memory_before_mb": gpu_info.get("memory_used_mb"),
        "gpu_memory_after_mb": None,
        "runtime_seconds": None,
        "status": status,
        "skip_reason": None,
        "error_message": None,
        "docling_version": docling_version(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "start_timestamp": utc_now(),
        "end_timestamp": None,
        "page_count": None,
        "number_of_tables_extracted": None,
        "table_metrics": empty_metrics(),
        "warnings": [],
        "output_file_paths": {},
    }


def row_from_metadata(metadata: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    metrics = dict(empty_metrics())
    metrics.update(metadata.get("table_metrics") or {})
    paths = metadata.get("output_file_paths") or {}
    row = {
        "pdf_filename": metadata.get("pdf_filename"),
        "profile": metadata.get("profile_name"),
        "profile_group": metadata.get("profile_group"),
        "quality_tier": metadata.get("quality_tier"),
        "resource_class": metadata.get("resource_class"),
        "status": metadata.get("status"),
        "skip_reason": metadata.get("skip_reason"),
        "error_message": metadata.get("error_message"),
        "runtime_seconds": metadata.get("runtime_seconds"),
        "pipeline": metadata.get("pipeline"),
        "pdf_backend": metadata.get("pdf_backend"),
        "table_mode": metadata.get("table_mode"),
        "do_cell_matching": metadata.get("do_cell_matching"),
        "ocr_engine": metadata.get("ocr_engine"),
        "ocr_lang": join_list(metadata.get("ocr_lang")),
        "psm": metadata.get("psm"),
        "do_ocr": metadata.get("do_ocr"),
        "force_full_page_ocr": metadata.get("force_full_page_ocr"),
        "vlm_model": metadata.get("vlm_model"),
        "images_scale": metadata.get("images_scale"),
        "generate_page_images": metadata.get("generate_page_images"),
        "generate_picture_images": metadata.get("generate_picture_images"),
        "device_requested": metadata.get("device_requested"),
        "device_selected": metadata.get("device_selected"),
        "gpu_available": metadata.get("gpu_available"),
        "gpu_name": metadata.get("gpu_name"),
        "gpu_memory_total_mb": metadata.get("gpu_memory_total_mb"),
        "gpu_memory_before_mb": metadata.get("gpu_memory_before_mb"),
        "gpu_memory_after_mb": metadata.get("gpu_memory_after_mb"),
        "page_count": metadata.get("page_count"),
        "table_count": metrics.get("table_count"),
        "markdown_output_path": paths.get("markdown"),
        "json_output_path": paths.get("json"),
        "run_dir": str(run_dir),
        "table_metrics_path": paths.get("table_metrics"),
        "inspection_path": paths.get("inspection"),
    }
    row.update({key: metrics.get(key) for key in TABLE_METRIC_KEYS})
    row["heuristic_score"] = score_candidate(row, SCORING_CONFIG)
    return row


def join_list(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(make_jsonable(metadata), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_profile(path: Path, profile: dict[str, Any]) -> None:
    import yaml

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(profile, handle, sort_keys=False, allow_unicode=True)


def append_errors(error_log: Path, text: str) -> None:
    if not text:
        return
    with error_log.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def run_one(
    pdf_path: Path,
    profile_name: str,
    profile: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
    api: dict[str, Any] | None,
    gpu_before: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    run_dir = output_root / pdf_path.stem / profile_name
    tables_dir = run_dir / "tables"
    images_dir = run_dir / "images"
    metadata_path = run_dir / "run_metadata.json"
    error_log = run_dir / "errors.log"
    device = selected_device(profile, args.device_override)

    if run_dir.exists() and not args.overwrite and metadata_path.exists():
        metadata = load_existing_metadata(metadata_path, pdf_path, profile_name, profile, device, gpu_before)
        metadata["status"] = "skipped"
        metadata["skip_reason"] = "existing_output_found"
        write_inspection(run_dir, profile_name, profile, metadata, metadata.get("table_metrics"), error_log)
        return row_from_metadata(metadata, run_dir), api

    run_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    error_log.write_text("", encoding="utf-8")
    write_profile(run_dir / "profile_used.yaml", {**profile, "selected_device": device})

    start_time = time.perf_counter()
    metadata = base_metadata(pdf_path, profile_name, profile, device, gpu_before, status="failure")
    metadata["output_file_paths"] = {
        "profile_used": str(run_dir / "profile_used.yaml"),
        "metadata": str(metadata_path),
        "inspection": str(run_dir / "inspection.md"),
        "tables_dir": str(tables_dir),
        "images_dir": str(images_dir),
    }

    try:
        if api is None:
            api = import_docling_api()

        context = BuildContext()
        pipeline_options = build_pipeline_options(profile, device, api, context)
        converter = build_converter(profile, pipeline_options, api)
        result, convert_warnings = convert_pdf(converter, pdf_path, args.max_pages)
        metadata["warnings"].extend(context.warnings)
        metadata["warnings"].extend(convert_warnings)
        doc = getattr(result, "document", result)

        output_paths = export_document(doc, run_dir)

        table_export = export_tables(doc, tables_dir)
        if table_export.get("errors"):
            metadata["warnings"].append("table_export_failed")
            append_errors(error_log, "\n".join(str(item) for item in table_export["errors"]))

        image_count = save_images(doc, images_dir)
        metrics = compute_table_metrics(tables_dir, run_dir / "output.md", run_dir / "output.json")
        if not metrics.get("table_shapes") and table_export.get("table_shapes"):
            metrics["table_shapes"] = table_export["table_shapes"]
        write_table_metrics(run_dir / "table_metrics.json", metrics)

        metadata["page_count"] = maybe_page_count(result, doc)
        metadata["number_of_tables_extracted"] = table_export.get("table_count")
        metadata["table_metrics"] = metrics
        metadata["output_file_paths"] = {
            **output_paths,
            "profile_used": str(run_dir / "profile_used.yaml"),
            "metadata": str(metadata_path),
            "table_metrics": str(run_dir / "table_metrics.json"),
            "inspection": str(run_dir / "inspection.md"),
            "tables_dir": str(tables_dir),
            "images_dir": str(images_dir),
            "images_saved": image_count,
        }
        metadata["status"] = "success"
    except UnsupportedFeature as exc:
        metadata["status"] = "skipped"
        metadata["skip_reason"] = exc.reason_code
        metadata["error_message"] = str(exc)
        append_errors(error_log, f"{exc.reason_code}: {exc}")
        LOGGER.warning("%s / %s skipped: %s", pdf_path.name, profile_name, exc.reason_code)
    except Exception as exc:
        metadata["status"] = "failure"
        metadata["error_message"] = str(exc)
        append_errors(error_log, traceback.format_exc())
        LOGGER.error("%s / %s failed: %s", pdf_path.name, profile_name, exc)
    finally:
        finalize_metadata(metadata, start_time, gpu_before)
        write_metadata(metadata_path, metadata)
        write_inspection(run_dir, profile_name, profile, metadata, metadata.get("table_metrics"), error_log)

    return row_from_metadata(metadata, run_dir), api


def load_existing_metadata(
    metadata_path: Path,
    pdf_path: Path,
    profile_name: str,
    profile: dict[str, Any],
    device: str,
    gpu_info: dict[str, Any],
) -> dict[str, Any]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        metadata = base_metadata(pdf_path, profile_name, profile, device, gpu_info, status="skipped")
    metadata.setdefault("profile_group", profile.get("group"))
    metadata.setdefault("quality_tier", profile.get("quality_tier"))
    metadata.setdefault("resource_class", profile.get("resource_class"))
    metadata.setdefault("table_metrics", compute_table_metrics(metadata_path.parent / "tables", metadata_path.parent / "output.md", metadata_path.parent / "output.json"))
    metadata.setdefault("output_file_paths", {})
    metadata["output_file_paths"].setdefault("inspection", str(metadata_path.parent / "inspection.md"))
    metadata["output_file_paths"].setdefault("metadata", str(metadata_path))
    return metadata


def finalize_metadata(metadata: dict[str, Any], start_time: float, gpu_before: dict[str, Any]) -> None:
    runtime = round(time.perf_counter() - start_time, 3)
    gpu_after = detect_gpu()
    metadata["end_timestamp"] = utc_now()
    metadata["runtime_seconds"] = runtime
    metadata["gpu_available"] = gpu_after.get("available", gpu_before.get("available"))
    metadata["gpu_name"] = gpu_after.get("name") or gpu_before.get("name")
    metadata["gpu_memory_total_mb"] = gpu_after.get("memory_total_mb") or gpu_before.get("memory_total_mb")
    metadata["gpu_memory_before_mb"] = gpu_before.get("memory_used_mb")
    metadata["gpu_memory_after_mb"] = gpu_after.get("memory_used_mb")
    metadata["gpu_memory_before"] = gpu_before
    metadata["gpu_memory_after"] = gpu_after


def terminal_row(
    pdf_path: Path,
    profile_name: str,
    profile: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
    gpu_info: dict[str, Any],
    status: str,
    reason: str,
) -> dict[str, Any]:
    run_dir = output_root / pdf_path.stem / profile_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(exist_ok=True)
    (run_dir / "images").mkdir(exist_ok=True)
    error_log = run_dir / "errors.log"
    error_log.write_text(reason + "\n", encoding="utf-8")

    device = selected_device(profile, args.device_override)
    write_profile(run_dir / "profile_used.yaml", {**profile, "selected_device": device})
    metadata = base_metadata(pdf_path, profile_name, profile, device, gpu_info, status=status)
    metadata["runtime_seconds"] = 0
    metadata["end_timestamp"] = utc_now()
    metadata["gpu_memory_after_mb"] = gpu_info.get("memory_used_mb")
    metadata["gpu_memory_before"] = gpu_info
    metadata["gpu_memory_after"] = gpu_info
    if status == "skipped":
        metadata["skip_reason"] = reason
    else:
        metadata["error_message"] = reason
    metadata["output_file_paths"] = {
        "profile_used": str(run_dir / "profile_used.yaml"),
        "metadata": str(run_dir / "run_metadata.json"),
        "inspection": str(run_dir / "inspection.md"),
        "tables_dir": str(run_dir / "tables"),
        "images_dir": str(run_dir / "images"),
    }
    write_metadata(run_dir / "run_metadata.json", metadata)
    write_inspection(run_dir, profile_name, profile, metadata, metadata.get("table_metrics"), error_log)
    return row_from_metadata(metadata, run_dir)


def write_inspection(
    run_dir: Path,
    profile_name: str,
    profile: dict[str, Any],
    metadata: dict[str, Any],
    metrics: dict[str, Any] | None,
    error_log: Path,
) -> None:
    errors_text = ""
    if error_log.exists():
        errors_text = error_log.read_text(encoding="utf-8", errors="replace")
    inspection = build_inspection_markdown(profile_name, profile, metadata, metrics, run_dir, errors_text)
    (run_dir / "inspection.md").write_text(inspection, encoding="utf-8")


def print_profile_list(profiles: dict[str, dict[str, Any]], names: list[str]) -> None:
    rows = profile_table_rows(profiles, names)
    print_table(
        rows,
        ["profile", "group", "quality_tier", "resource_class", "gpu_required", "gpu_preferred", "description"],
    )


def print_dry_run_plan(
    pdfs: list[Path],
    profiles: dict[str, dict[str, Any]],
    names: list[str],
    args: argparse.Namespace,
    output_root: Path,
    gpu_info: dict[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
    for pdf_path in pdfs:
        for name in names:
            profile = profiles[name]
            reason = plan_skip_reason(
                profile,
                args.allow_gpu,
                args.allow_missing_gpu,
                bool(gpu_info.get("available")),
                args.device_override,
            )
            if output_exists(output_root, pdf_path, name) and not args.overwrite:
                reason = reason or "existing_output_found"
            rows.append(
                {
                    "pdf": pdf_path.name,
                    "profile": name,
                    "group": profile.get("group"),
                    "tier": profile.get("quality_tier"),
                    "resource": profile.get("resource_class"),
                    "device": selected_device(profile, args.device_override),
                    "plan": "skip" if reason else "run",
                    "reason": reason or "",
                }
            )
    print_table(rows, ["pdf", "profile", "group", "tier", "resource", "device", "plan", "reason"])


def print_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        print("No rows.")
        return
    try:
        from tabulate import tabulate

        print(tabulate([[row.get(col, "") for col in columns] for row in rows], headers=columns, tablefmt="github"))
        return
    except Exception:
        pass
    print(dataframe_to_markdown(rows, columns))


def write_all_summaries(rows: list[dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    write_report(rows, SUMMARY_COLUMNS, output_root / "summary", markdown_prefix=None)

    table_rows = [{column: row.get(column) for column in TABLE_SUMMARY_COLUMNS} for row in rows]
    write_report(table_rows, TABLE_SUMMARY_COLUMNS, output_root / "table_quality_summary", markdown_prefix=None)

    best_rows = []
    for row in rows:
        candidate = {column: row.get(column) for column in BEST_CANDIDATE_COLUMNS}
        candidate["heuristic_score"] = score_candidate(row, SCORING_CONFIG)
        best_rows.append(candidate)
    best_rows.sort(
        key=lambda item: (
            str(item.get("pdf_filename") or ""),
            -float(item.get("heuristic_score") or 0),
            str(item.get("profile") or ""),
        )
    )
    warning = (
        "This ranking is heuristic. The JSON and table CSV/HTML outputs must be inspected manually "
        "before choosing the final extraction profile.\n\n"
    )
    write_report(best_rows, BEST_CANDIDATE_COLUMNS, output_root / "best_candidates", markdown_prefix=warning)


def write_report(
    rows: list[dict[str, Any]],
    columns: list[str],
    stem: Path,
    markdown_prefix: str | None = None,
) -> None:
    try:
        import pandas as pd

        df = pd.DataFrame(rows, columns=columns).fillna("")
        df.to_csv(stem.with_suffix(".csv"), index=False)
        df.to_excel(stem.with_suffix(".xlsx"), index=False)
        try:
            markdown = df.to_markdown(index=False)
        except Exception:
            markdown = dataframe_to_markdown(rows, columns)
        stem.with_suffix(".md").write_text((markdown_prefix or "") + markdown + "\n", encoding="utf-8")
    except Exception as exc:
        LOGGER.warning("Falling back to standard-library writer for %s: %s", stem.name, exc)
        write_csv_summary(rows, columns, stem.with_suffix(".csv"))
        stem.with_suffix(".md").write_text(
            (markdown_prefix or "") + dataframe_to_markdown(rows, columns) + "\n",
            encoding="utf-8",
        )
        write_minimal_xlsx(rows, columns, stem.with_suffix(".xlsx"))


def write_csv_summary(rows: list[dict[str, Any]], columns: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: stringify_cell(row.get(column)) for column in columns})


def dataframe_to_markdown(rows: list[dict[str, Any]], columns: list[str]) -> str:
    values = [[stringify_cell(row.get(column)) for column in columns] for row in rows]
    widths = [max([len(column)] + [len(row[index]) for row in values]) for index, column in enumerate(columns)]
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
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
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


def validate_args(args: argparse.Namespace) -> None:
    if not args.list_profiles and not args.input:
        raise SystemExit("--input is required unless --list-profiles is used.")
    if args.allow_missing_gpu and not args.allow_gpu:
        LOGGER.warning("--allow-missing-gpu has no effect unless --allow-gpu is also passed.")


def main() -> int:
    setup_logging()
    args = parse_args()
    validate_args(args)

    config_path = Path(args.config).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    profiles = load_profiles(config_path)
    selected_profile_names = select_profile_names(
        profiles,
        args.profiles,
        args.profile_groups,
        args.quality_tiers,
        args.resource_classes,
    )

    if args.list_profiles:
        print_profile_list(profiles, selected_profile_names)
        return 0

    input_path = Path(args.input).expanduser().resolve()
    pdfs = find_pdfs(input_path)
    if not pdfs:
        LOGGER.warning("No PDFs found under %s", input_path)

    gpu_info = detect_gpu()
    LOGGER.info(
        "GPU available: %s%s",
        gpu_info.get("available"),
        f" ({gpu_info.get('name')})" if gpu_info.get("name") else "",
    )

    if args.dry_run_plan:
        print_dry_run_plan(pdfs, profiles, selected_profile_names, args, output_root, gpu_info)
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    api: dict[str, Any] | None = None
    total = len(pdfs) * len(selected_profile_names)

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = SimpleProgress  # type: ignore[assignment]

    with tqdm(total=total, desc="Docling ablation", unit="run") as progress:
        for pdf_path in pdfs:
            for profile_name in selected_profile_names:
                profile = normalize_profile(profile_name, profiles[profile_name])
                device = selected_device(profile, args.device_override)
                profile["device"] = device

                reason = plan_skip_reason(
                    profile,
                    args.allow_gpu,
                    args.allow_missing_gpu,
                    bool(gpu_info.get("available")),
                    args.device_override,
                )
                try:
                    if reason == "gpu_profile_not_allowed":
                        rows.append(
                            terminal_row(
                                pdf_path,
                                profile_name,
                                profile,
                                args,
                                output_root,
                                gpu_info,
                                "skipped",
                                "gpu_profile_not_allowed",
                            )
                        )
                    elif reason == "gpu_unavailable":
                        rows.append(
                            terminal_row(
                                pdf_path,
                                profile_name,
                                profile,
                                args,
                                output_root,
                                gpu_info,
                                "skipped",
                                "gpu_unavailable",
                            )
                        )
                    elif reason == "gpu_unavailable_error":
                        rows.append(
                            terminal_row(
                                pdf_path,
                                profile_name,
                                profile,
                                args,
                                output_root,
                                gpu_info,
                                "failure",
                                "gpu_unavailable",
                            )
                        )
                    else:
                        row, api = run_one(pdf_path, profile_name, profile, args, output_root, api, gpu_info)
                        rows.append(row)
                finally:
                    progress.update(1)

    write_all_summaries(rows, output_root)
    LOGGER.info("Wrote summary files to %s", output_root)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
