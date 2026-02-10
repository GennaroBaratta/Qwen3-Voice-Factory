from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    from ._common import ROOT, run, venv_python_path
except ImportError:
    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from _common import ROOT, run, venv_python_path


PATCH_SCRIPT = ROOT / "scripts" / "patch_qwen_tts_v1_removal.py"
APP_FILE = ROOT / "app.py"


def run_capture_stdout(cmd: list[str], *, cwd: Path | None = None) -> str:
    print("+", " ".join(cmd))
    completed = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def preflight_runtime_imports(python_exec: Path) -> bool:
    check = (
        "import importlib\n"
        "import sys\n"
        "for module_name in ('torch', 'torchvision', 'torchaudio', 'qwen_tts'):\n"
        "    try:\n"
        "        importlib.import_module(module_name)\n"
        "    except Exception as exc:\n"
        "        print(f'ERROR: Runtime stack is inconsistent ({module_name}): {exc}', file=sys.stderr)\n"
        "        print('Run: python scripts/install.py', file=sys.stderr)\n"
        "        sys.exit(1)\n"
        "import torch\n"
        "print(f'torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}')\n"
    )
    try:
        run([str(python_exec), "-c", check])
        return True
    except subprocess.CalledProcessError:
        return False


def detect_runtime_device(python_exec: Path) -> str:
    probe = "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
    output = run_capture_stdout([str(python_exec), "-c", probe])
    device = output.splitlines()[-1].strip().lower() if output else ""
    if device not in {"cuda", "cpu"}:
        raise RuntimeError(f"Unexpected runtime device probe output: {output!r}")
    return device


def main() -> int:
    venv_python = venv_python_path(ROOT)
    if not venv_python.exists():
        print("ERROR: Python environment not found!", file=sys.stderr)
        print("Run: python scripts/install.py", file=sys.stderr)
        return 1

    try:
        run([str(venv_python), str(PATCH_SCRIPT)])
        if not preflight_runtime_imports(venv_python):
            return 1

        detected_device = detect_runtime_device(venv_python)
        requested_device = os.environ.get("QWEN_DEVICE", "").strip().lower()

        if requested_device in {"cpu", "cuda"}:
            runtime_device = requested_device
        else:
            runtime_device = detected_device
            if requested_device:
                print(
                    f"WARNING: Invalid QWEN_DEVICE='{requested_device}', falling back to auto detection.",
                    file=sys.stderr,
                )

        if runtime_device == "cuda" and detected_device != "cuda":
            print("ERROR: QWEN_DEVICE=cuda requested but CUDA is not available.", file=sys.stderr)
            return 1

        if runtime_device == "cuda":
            print("Runtime mode: CUDA")
        else:
            if requested_device == "cpu" and detected_device == "cuda":
                print("Runtime mode: CPU (forced by QWEN_DEVICE=cpu)")
            else:
                print("Runtime mode: CPU")
                print("WARNING: CUDA not detected. CPU mode is supported but significantly slower.")

        launch_env = os.environ.copy()
        launch_env["QWEN_DEVICE"] = runtime_device
        run([str(venv_python), str(APP_FILE)], env=launch_env)
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
