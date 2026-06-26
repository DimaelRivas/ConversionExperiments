"""GPU/environment helpers for the Docling ablation runner."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any


def safe_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def run_command(command: list[str], timeout: int = 10) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        text = (result.stdout or result.stderr or "").strip()
        return result.returncode == 0, text
    except Exception as exc:
        return False, str(exc)


def detect_gpu() -> dict[str, Any]:
    """Return best-effort NVIDIA GPU details without requiring GPU packages."""

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
                info["memory_total_mb"] = safe_int(parts[1])
                info["memory_used_mb"] = safe_int(parts[2])

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
        try:
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
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass

    info["sources"] = sorted(set(info["sources"]))
    return info
