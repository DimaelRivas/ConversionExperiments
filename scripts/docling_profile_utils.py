"""Profile normalization, filtering, and display helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_PROFILE_METADATA: dict[str, Any] = {
    "description": "",
    "group": "baseline",
    "quality_tier": "baseline",
    "gpu_required": False,
    "gpu_preferred": False,
    "resource_class": "low",
}


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_profile(name: str, profile: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(DEFAULT_PROFILE_METADATA)
    normalized.update(profile or {})
    normalized.setdefault("description", name)
    normalized["gpu_required"] = bool(normalized.get("gpu_required", False))
    normalized["gpu_preferred"] = bool(normalized.get("gpu_preferred", False))
    normalized["group"] = str(normalized.get("group") or "baseline")
    normalized["quality_tier"] = str(normalized.get("quality_tier") or "baseline")
    normalized["resource_class"] = str(normalized.get("resource_class") or "low")
    return normalized


def normalize_profiles(profiles: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: normalize_profile(name, profile) for name, profile in profiles.items()}


def select_profile_names(
    profiles: dict[str, dict[str, Any]],
    profiles_spec: str | None,
    group_spec: str | None = None,
    tier_spec: str | None = None,
    resource_spec: str | None = None,
) -> list[str]:
    spec = (profiles_spec or "all").strip()
    if spec.lower() == "all":
        selected = list(profiles.keys())
    else:
        selected = split_csv(spec)
        missing = [name for name in selected if name not in profiles]
        if missing:
            raise ValueError(f"Unknown profile(s): {', '.join(missing)}")

    groups = set(split_csv(group_spec))
    tiers = set(split_csv(tier_spec))
    resources = set(split_csv(resource_spec))

    filtered: list[str] = []
    for name in selected:
        profile = normalize_profile(name, profiles[name])
        profile_groups = {str(profile.get("group"))}
        aliases = profile.get("group_aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        profile_groups.update(str(alias) for alias in aliases)
        if groups and not profile_groups.intersection(groups):
            continue
        if tiers and profile.get("quality_tier") not in tiers:
            continue
        if resources and profile.get("resource_class") not in resources:
            continue
        filtered.append(name)
    return filtered


def profile_table_rows(profiles: dict[str, dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in names:
        profile = normalize_profile(name, profiles[name])
        rows.append(
            {
                "profile": name,
                "group": profile.get("group"),
                "quality_tier": profile.get("quality_tier"),
                "resource_class": profile.get("resource_class"),
                "gpu_required": profile.get("gpu_required"),
                "gpu_preferred": profile.get("gpu_preferred"),
                "description": profile.get("description"),
            }
        )
    return rows


def selected_device(profile: dict[str, Any], device_override: str | None = None) -> str:
    return str(device_override or profile.get("device") or "auto").lower()


def output_exists(output_root: Path, pdf_path: Path, profile_name: str) -> bool:
    return (output_root / pdf_path.stem / profile_name / "run_metadata.json").exists()


def plan_skip_reason(
    profile: dict[str, Any],
    allow_gpu: bool,
    allow_missing_gpu: bool,
    gpu_available: bool,
    device_override: str | None = None,
) -> str | None:
    device = selected_device(profile, device_override)
    needs_gpu = bool(profile.get("gpu_required", False)) or device == "cuda"
    if needs_gpu and not allow_gpu:
        return "gpu_profile_not_allowed"
    if needs_gpu and allow_gpu and not gpu_available:
        if allow_missing_gpu:
            return "gpu_unavailable"
        return "gpu_unavailable_error"
    return None
