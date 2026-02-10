from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
UV_INSTALL_DOCS = "https://docs.astral.sh/uv/getting-started/installation/"


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd or ROOT), env=dict(env) if env else None, check=True)


def venv_python_path(root: Path = ROOT) -> Path:
    venv_dir = root / ".venv"
    windows_python = venv_dir / "Scripts" / "python.exe"
    posix_python = venv_dir / "bin" / "python"

    candidates = [windows_python, posix_python] if os.name == "nt" else [posix_python, windows_python]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def find_uv() -> str | None:
    return shutil.which("uv")


def require_uv() -> str:
    uv = find_uv()
    if uv:
        return uv

    raise RuntimeError(
        "uv is not installed or not available on PATH. "
        f"Install uv first: {UV_INSTALL_DOCS}"
    )


def has_nvidia_smi() -> bool:
    if platform.system().lower() not in {"windows", "linux"}:
        return False

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False

    return result.returncode == 0
