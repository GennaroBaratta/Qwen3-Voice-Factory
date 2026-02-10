from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Literal

try:
    from ._common import ROOT, has_nvidia_smi, require_uv, run, venv_python_path
except ImportError:
    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from _common import ROOT, has_nvidia_smi, require_uv, run, venv_python_path


TorchProfile = Literal["cuda_nightly", "cpu_stable"]
PATCH_SCRIPT = ROOT / "scripts" / "patch_qwen_tts_v1_removal.py"
NUMPY_COMPAT_SPEC = "numpy<2.4"


def ensure_venv(uv: str) -> Path:
    python_exec = venv_python_path(ROOT)
    if python_exec.exists():
        return python_exec

    print("[1/6] Creating virtual environment (Python 3.11)")
    run([uv, "venv", ".venv", "--python", "3.11"])

    python_exec = venv_python_path(ROOT)
    if not python_exec.exists():
        raise RuntimeError("Virtual environment was created but Python executable was not found.")
    return python_exec


def sync_project(uv: str, python_exec: Path) -> None:
    print("[2/6] Installing project dependencies")
    run([uv, "sync", "--python", str(python_exec), "--no-dev"])


def detect_torch_profile() -> TorchProfile:
    os_name = platform.system().lower()
    if os_name in {"windows", "linux"} and has_nvidia_smi():
        return "cuda_nightly"
    return "cpu_stable"


def install_torch_cuda_nightly(uv: str, python_exec: Path) -> None:
    print("[3/6] Installing PyTorch Nightly CUDA profile")
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python_exec),
            "--upgrade",
            "--reinstall-package",
            "torch",
            "--reinstall-package",
            "torchvision",
            "--reinstall-package",
            "torchaudio",
            "--pre",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/nightly/cu128",
        ]
    )


def install_torch_cpu_stable(uv: str, python_exec: Path) -> None:
    print("[3/6] Installing PyTorch stable CPU profile")
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python_exec),
            "--upgrade",
            "--reinstall-package",
            "torch",
            "--reinstall-package",
            "torchvision",
            "--reinstall-package",
            "torchaudio",
            "--torch-backend",
            "cpu",
            "torch",
            "torchvision",
            "torchaudio",
        ]
    )


def pin_numpy_compat(uv: str, python_exec: Path) -> None:
    print("[4/6] Pinning NumPy for numba/librosa compatibility")
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python_exec),
            "--upgrade",
            "--reinstall-package",
            "numpy",
            NUMPY_COMPAT_SPEC,
        ]
    )


def verify_torch_stack(python_exec: Path, expected_profile: TorchProfile) -> bool:
    check = (
        "import importlib\n"
        "import sys\n"
        "expected_profile = sys.argv[1]\n"
        "for module_name in ('torch', 'torchvision', 'torchaudio', 'qwen_tts'):\n"
        "    try:\n"
        "        importlib.import_module(module_name)\n"
        "    except Exception as exc:\n"
        "        print(f'ERROR: Failed importing {module_name}: {exc}', file=sys.stderr)\n"
        "        sys.exit(1)\n"
        "import torch\n"
        "cuda_available = torch.cuda.is_available()\n"
        "print(f'torch={torch.__version__} cuda={torch.version.cuda} available={cuda_available}')\n"
        "if expected_profile == 'cuda_nightly' and not cuda_available:\n"
        "    print('ERROR: CUDA was expected but is not available.', file=sys.stderr)\n"
        "    sys.exit(2)\n"
    )
    try:
        run([str(python_exec), "-c", check, expected_profile])
        return True
    except subprocess.CalledProcessError:
        return False


def apply_patch(python_exec: Path) -> None:
    print("[5/6] Applying qwen-tts 12Hz-only patch")
    run([str(python_exec), str(PATCH_SCRIPT)])


def main() -> int:
    try:
        uv = require_uv()
        python_exec = ensure_venv(uv)
        sync_project(uv, python_exec)

        selected_profile = detect_torch_profile()
        print(f"Detected torch profile: {selected_profile}")

        if selected_profile == "cuda_nightly":
            install_torch_cuda_nightly(uv, python_exec)
        else:
            install_torch_cpu_stable(uv, python_exec)
        pin_numpy_compat(uv, python_exec)

        apply_patch(python_exec)

        print("[6/6] Verifying runtime stack")
        if not verify_torch_stack(python_exec, selected_profile):
            if selected_profile == "cuda_nightly":
                print("CUDA verification failed; retrying with CPU stable profile.")
                selected_profile = "cpu_stable"
                install_torch_cpu_stable(uv, python_exec)
                pin_numpy_compat(uv, python_exec)
                apply_patch(python_exec)
                if not verify_torch_stack(python_exec, selected_profile):
                    raise RuntimeError("Torch stack verification failed after CPU fallback.")
            else:
                raise RuntimeError("Torch stack verification failed.")

        print(f"Installation complete with profile: {selected_profile}")
        print("Run: python scripts/start.py")
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
