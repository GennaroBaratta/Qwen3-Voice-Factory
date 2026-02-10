from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PATCH_SCRIPT = ROOT / "scripts" / "patch_qwen_tts_v1_removal.py"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd or ROOT), check=True)


def resolve_uv() -> str | None:
    uv = shutil.which("uv")
    if uv:
        return uv

    user_home = Path(os.environ.get("USERPROFILE", ""))
    candidates = [
        user_home / ".local" / "bin" / "uv.exe",
        user_home / ".cargo" / "bin" / "uv.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def ensure_uv() -> str:
    uv = resolve_uv()
    if uv:
        return uv

    print("[1/5] Installing uv (Python manager)")
    run(
        [
            "powershell",
            "-ExecutionPolicy",
            "ByPass",
            "-c",
            "irm https://astral.sh/uv/install.ps1 | iex",
        ]
    )

    uv = resolve_uv()
    if not uv:
        raise RuntimeError("uv not found after installation. Restart terminal or install uv manually.")
    return uv


def main() -> int:
    try:
        uv = ensure_uv()

        if not VENV_PYTHON.exists():
            print("[2/5] Creating virtual environment (Python 3.11)")
            run([uv, "venv", ".venv", "--python", "3.11"])

        print("[3/5] Installing PyTorch Nightly (Blackwell Support)")
        run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(VENV_PYTHON),
                "--pre",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/nightly/cu128",
            ]
        )

        print("[4/5] Installing project dependencies")
        run([uv, "sync", "--python", str(VENV_PYTHON), "--no-dev"])

        print("[5/5] Applying qwen-tts 12Hz-only patch")
        run([str(VENV_PYTHON), str(PATCH_SCRIPT)])

        print("Installation complete. Run: python scripts/start.py")
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
