from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PATCH_SCRIPT = ROOT / "scripts" / "patch_qwen_tts_v1_removal.py"
APP_FILE = ROOT / "app.py"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd or ROOT), check=True)


def main() -> int:
    if not VENV_PYTHON.exists():
        print("ERROR: Python environment not found!", file=sys.stderr)
        print("Run: python scripts/install.py", file=sys.stderr)
        return 1

    try:
        run([str(VENV_PYTHON), str(PATCH_SCRIPT)])
        run([str(VENV_PYTHON), str(APP_FILE)])
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
