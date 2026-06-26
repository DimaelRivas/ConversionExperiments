"""Best-effort Docling table export, metrics, inspections, and scoring."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


TABLE_METRIC_KEYS = [
    "table_count",
    "tables_with_rows",
    "total_rows",
    "total_columns_sum",
    "max_columns",
    "max_rows",
    "empty_cell_ratio",
    "single_column_table_count",
    "very_small_table_count",
    "csv_files_count",
    "html_files_count",
    "markdown_table_pipe_count",
    "json_table_item_count",
]

SCORING_CONFIG: dict[str, float] = {
    "success_base": 100.0,
    "table_count_weight": 10.0,
    "tables_with_rows_weight": 5.0,
    "total_rows_weight": 0.4,
    "total_columns_weight": 0.8,
    "max_columns_weight": 2.0,
    "empty_cell_ratio_penalty": 35.0,
    "single_column_table_penalty": 5.0,
    "very_small_table_penalty": 3.0,
    "runtime_penalty_per_second": 0.005,
}


def empty_metrics() -> dict[str, Any]:
    metrics = {key: 0 for key in TABLE_METRIC_KEYS}
    metrics["empty_cell_ratio"] = 0.0
    metrics["table_shapes"] = []
    return metrics


def export_tables(doc: Any, tables_dir: Path) -> dict[str, Any]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    tables = collect_table_candidates(doc)
    exported = 0
    errors: list[str] = []
    shapes: list[dict[str, int]] = []

    for index, table in enumerate(tables, start=1):
        stem = tables_dir / f"table_{index:03d}"
        try:
            dataframe = table_to_dataframe(table, doc)
            html = table_to_html(table, doc)

            if dataframe is not None:
                dataframe.to_csv(stem.with_suffix(".csv"), index=False)
                try:
                    dataframe.to_html(stem.with_suffix(".html"), index=False)
                except Exception:
                    if html:
                        stem.with_suffix(".html").write_text(html, encoding="utf-8")
                exported += 1
                shapes.append({"rows": int(getattr(dataframe, "shape", [0, 0])[0]), "columns": int(getattr(dataframe, "shape", [0, 0])[1])})
                continue

            if html:
                stem.with_suffix(".html").write_text(html, encoding="utf-8")
                csv_from_html = html_to_dataframe(html)
                if csv_from_html is not None:
                    csv_from_html.to_csv(stem.with_suffix(".csv"), index=False)
                    shapes.append({"rows": int(csv_from_html.shape[0]), "columns": int(csv_from_html.shape[1])})
                exported += 1
                continue

            errors.append(f"table_{index:03d}: no supported dataframe/html export API")
        except Exception as exc:
            errors.append(f"table_{index:03d}: {exc}")

    return {"table_count": exported, "errors": errors, "table_shapes": shapes}


def collect_table_candidates(doc: Any) -> list[Any]:
    seen: set[int] = set()
    tables: list[Any] = []

    def add(candidate: Any) -> None:
        if candidate is None:
            return
        ident = id(candidate)
        if ident in seen:
            return
        seen.add(ident)
        tables.append(candidate)

    collection = getattr(doc, "tables", None)
    if isinstance(collection, dict):
        for item in collection.values():
            add(item)
    elif collection:
        for item in collection:
            add(item)

    iterator = getattr(doc, "iterate_items", None)
    if callable(iterator):
        try:
            for entry in iterator():
                item = entry[0] if isinstance(entry, tuple) and entry else entry
                label = str(getattr(item, "label", "")).lower()
                class_name = item.__class__.__name__.lower()
                if "table" in label or "table" in class_name:
                    add(item)
        except Exception:
            pass

    return tables


def table_to_dataframe(table: Any, doc: Any) -> Any | None:
    try:
        import pandas as pd
    except Exception:
        pd = None  # type: ignore[assignment]

    for target in [table, getattr(table, "data", None)]:
        if target is None:
            continue
        for name in ["export_to_dataframe", "to_dataframe", "as_dataframe"]:
            method = getattr(target, name, None)
            dataframe = call_docling_export(method, doc)
            if dataframe is not None:
                return dataframe

    data = getattr(table, "data", None)
    if pd is not None and isinstance(data, list):
        return pd.DataFrame(data)

    dataframe = table_data_to_dataframe(data)
    if dataframe is not None:
        return dataframe
    return None


def table_data_to_dataframe(data: Any) -> Any | None:
    if data is None:
        return None
    try:
        import pandas as pd
    except Exception:
        return None

    cells = getattr(data, "table_cells", None) or getattr(data, "cells", None)
    if not cells:
        return None

    row_count = int(getattr(data, "num_rows", 0) or getattr(data, "row_count", 0) or 0)
    col_count = int(getattr(data, "num_cols", 0) or getattr(data, "num_columns", 0) or getattr(data, "col_count", 0) or 0)
    for cell in cells:
        row = first_int_attr(cell, ["start_row_offset_idx", "row_idx", "row", "row_index"])
        col = first_int_attr(cell, ["start_col_offset_idx", "col_idx", "column", "column_index", "col"])
        row_span = max(1, first_int_attr(cell, ["row_span", "rowspan"]) or 1)
        col_span = max(1, first_int_attr(cell, ["col_span", "colspan"]) or 1)
        if row is not None:
            row_count = max(row_count, row + row_span)
        if col is not None:
            col_count = max(col_count, col + col_span)

    if row_count <= 0 or col_count <= 0:
        return None

    grid = [["" for _ in range(col_count)] for _ in range(row_count)]
    for cell in cells:
        row = first_int_attr(cell, ["start_row_offset_idx", "row_idx", "row", "row_index"]) or 0
        col = first_int_attr(cell, ["start_col_offset_idx", "col_idx", "column", "column_index", "col"]) or 0
        text = str(getattr(cell, "text", "") or getattr(cell, "content", "") or getattr(cell, "value", "") or "")
        if 0 <= row < row_count and 0 <= col < col_count:
            grid[row][col] = text
    return pd.DataFrame(grid)


def first_int_attr(obj: Any, names: list[str]) -> int | None:
    for name in names:
        value = getattr(obj, name, None)
        try:
            if value is not None:
                return int(value)
        except Exception:
            continue
    return None


def table_to_html(table: Any, doc: Any) -> str | None:
    for target in [table, getattr(table, "data", None)]:
        if target is None:
            continue
        for name in ["export_to_html", "to_html", "as_html"]:
            method = getattr(target, name, None)
            html = call_docling_export(method, doc)
            if html:
                return str(html)
    return None


def call_docling_export(method: Any, doc: Any) -> Any | None:
    if not callable(method):
        return None
    attempts = [
        ((), {"doc": doc}),
        ((), {"document": doc}),
        ((doc,), {}),
        ((), {}),
    ]
    for args, kwargs in attempts:
        try:
            return method(*args, **kwargs)
        except TypeError:
            continue
    return None


def html_to_dataframe(html: str) -> Any | None:
    try:
        import pandas as pd

        frames = pd.read_html(html)
        return frames[0] if frames else None
    except Exception:
        return None


def compute_table_metrics(tables_dir: Path, markdown_path: Path | None, json_path: Path | None) -> dict[str, Any]:
    metrics = empty_metrics()
    csv_paths = sorted(tables_dir.glob("*.csv")) if tables_dir.exists() else []
    html_paths = sorted(tables_dir.glob("*.html")) if tables_dir.exists() else []

    metrics["csv_files_count"] = len(csv_paths)
    metrics["html_files_count"] = len(html_paths)

    total_cells = 0
    empty_cells = 0
    shapes: list[dict[str, int]] = []
    for path in csv_paths:
        rows = read_csv_rows(path)
        if not rows:
            shapes.append({"rows": 0, "columns": 0})
            continue
        columns = max((len(row) for row in rows), default=0)
        row_count = len(rows)
        shapes.append({"rows": row_count, "columns": columns})
        metrics["total_rows"] += row_count
        metrics["total_columns_sum"] += columns
        metrics["max_rows"] = max(metrics["max_rows"], row_count)
        metrics["max_columns"] = max(metrics["max_columns"], columns)
        if row_count > 0:
            metrics["tables_with_rows"] += 1
        if columns <= 1:
            metrics["single_column_table_count"] += 1
        if row_count <= 2 and columns <= 2:
            metrics["very_small_table_count"] += 1
        for row in rows:
            total_cells += columns
            empty_cells += sum(1 for value in row + [""] * max(0, columns - len(row)) if not str(value).strip())

    metrics["json_table_item_count"] = count_json_tables(json_path)
    metrics["markdown_table_pipe_count"] = count_markdown_table_lines(markdown_path)
    metrics["table_count"] = max(
        len(csv_paths),
        len(html_paths),
        int(metrics["json_table_item_count"]),
    )
    metrics["empty_cell_ratio"] = round(empty_cells / total_cells, 4) if total_cells else 0.0
    metrics["table_shapes"] = shapes
    return metrics


def read_csv_rows(path: Path) -> list[list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [row for row in csv.reader(handle)]
    except UnicodeDecodeError:
        with path.open("r", encoding="latin-1", newline="") as handle:
            return [row for row in csv.reader(handle)]
    except Exception:
        return []


def count_markdown_table_lines(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
            count += 1
    return count


def count_json_tables(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    tables = data.get("tables") if isinstance(data, dict) else None
    if isinstance(tables, list):
        return len(tables)
    return count_table_labels(data)


def count_table_labels(value: Any) -> int:
    if isinstance(value, dict):
        label = str(value.get("label", "")).lower()
        own = 1 if label == "table" else 0
        return own + sum(count_table_labels(item) for item in value.values())
    if isinstance(value, list):
        return sum(count_table_labels(item) for item in value)
    return 0


def write_table_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def score_candidate(row: dict[str, Any], scoring: dict[str, float] | None = None) -> float:
    scoring = scoring or SCORING_CONFIG
    if row.get("status") != "success":
        return 0.0

    score = scoring["success_base"]
    score += numeric(row.get("table_count")) * scoring["table_count_weight"]
    score += numeric(row.get("tables_with_rows")) * scoring["tables_with_rows_weight"]
    score += numeric(row.get("total_rows")) * scoring["total_rows_weight"]
    score += numeric(row.get("total_columns_sum")) * scoring["total_columns_weight"]
    score += numeric(row.get("max_columns")) * scoring["max_columns_weight"]
    score -= numeric(row.get("empty_cell_ratio")) * scoring["empty_cell_ratio_penalty"]
    score -= numeric(row.get("single_column_table_count")) * scoring["single_column_table_penalty"]
    score -= numeric(row.get("very_small_table_count")) * scoring["very_small_table_penalty"]

    if row.get("pipeline") != "vlm" and row.get("profile_group") != "vlm":
        score -= numeric(row.get("runtime_seconds")) * scoring["runtime_penalty_per_second"]

    return round(score, 4)


def numeric(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def first_markdown_lines(path: Path, limit: int = 20) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[:limit]


def build_inspection_markdown(
    profile_name: str,
    profile: dict[str, Any],
    metadata: dict[str, Any],
    metrics: dict[str, Any] | None,
    run_dir: Path,
    errors_text: str = "",
) -> str:
    metrics = metrics or empty_metrics()
    markdown_path = run_dir / "output.md"
    csv_paths = sorted((run_dir / "tables").glob("*.csv"))
    top_shapes = sorted(
        metrics.get("table_shapes", []),
        key=lambda item: (int(item.get("rows", 0)), int(item.get("columns", 0))),
        reverse=True,
    )[:10]

    lines = [
        f"# Inspection: {profile_name}",
        "",
        f"- Status: {metadata.get('status')}",
        f"- Runtime seconds: {metadata.get('runtime_seconds')}",
        f"- Profile group: {metadata.get('profile_group')}",
        f"- Quality tier: {metadata.get('quality_tier')}",
        f"- Resource class: {metadata.get('resource_class')}",
        f"- Pipeline: {metadata.get('pipeline')}",
        f"- Device requested: {metadata.get('device_requested')}",
        f"- Device selected: {metadata.get('device_selected')}",
        f"- Tables: {metrics.get('table_count', 0)}",
        "",
        "## Table CSV files",
    ]

    if csv_paths:
        lines.extend(f"- {path.as_posix()}" for path in csv_paths)
    else:
        lines.append("- None")

    lines.extend(["", "## Top table shapes"])
    if top_shapes:
        for shape in top_shapes:
            lines.append(f"- rows={shape.get('rows', 0)}, columns={shape.get('columns', 0)}")
    else:
        lines.append("- None")

    lines.extend(["", "## First 20 Markdown Lines", "", "```markdown"])
    lines.extend(first_markdown_lines(markdown_path, 20))
    lines.append("```")

    skip_reason = metadata.get("skip_reason")
    error_message = metadata.get("error_message")
    warnings = metadata.get("warnings") or []
    if skip_reason or error_message or errors_text or warnings:
        lines.extend(["", "## Errors Or Skips"])
        if skip_reason:
            lines.append(f"- Skip reason: {skip_reason}")
        if error_message:
            lines.append(f"- Error: {error_message}")
        for warning in warnings:
            lines.append(f"- Warning: {warning}")
        if errors_text:
            compact = re.sub(r"\n{3,}", "\n\n", errors_text.strip())
            lines.extend(["", "```text", compact[:5000], "```"])

    return "\n".join(lines).rstrip() + "\n"
