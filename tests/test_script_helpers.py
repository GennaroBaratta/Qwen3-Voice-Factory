from __future__ import annotations

import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import scripts._common as common
import scripts.install as install
import scripts.start as start

class VenvPythonPathTests(unittest.TestCase):
    def test_venv_python_path_prefers_windows_on_nt(self) -> None:
        root = Path("C:/virtual-project")
        windows_python = root / ".venv" / "Scripts" / "python.exe"
        posix_python = root / ".venv" / "bin" / "python"

        with patch("scripts._common.os.name", "nt"):
            with patch.object(Path, "exists", autospec=True) as mocked_exists:
                mocked_exists.side_effect = lambda candidate: candidate in {windows_python, posix_python}
                resolved = common.venv_python_path(root)

        self.assertEqual(resolved, windows_python)

    def test_venv_python_path_prefers_posix_on_posix(self) -> None:
        root = Path("/virtual-project")
        windows_python = root / ".venv" / "Scripts" / "python.exe"
        posix_python = root / ".venv" / "bin" / "python"

        with patch("scripts._common.os.name", "posix"):
            with patch.object(Path, "exists", autospec=True) as mocked_exists:
                mocked_exists.side_effect = lambda candidate: candidate in {windows_python, posix_python}
                resolved = common.venv_python_path(root)

        self.assertEqual(resolved, posix_python)

    def test_venv_python_path_falls_back_to_other_existing_path(self) -> None:
        root = Path("C:/virtual-project")
        windows_python = root / ".venv" / "Scripts" / "python.exe"
        posix_python = root / ".venv" / "bin" / "python"

        with patch("scripts._common.os.name", "nt"):
            with patch.object(Path, "exists", autospec=True) as mocked_exists:
                mocked_exists.side_effect = lambda candidate: candidate == posix_python
                resolved = common.venv_python_path(root)

        self.assertNotEqual(resolved, windows_python)
        self.assertEqual(resolved, posix_python)


class UvResolutionTests(unittest.TestCase):
    def test_require_uv_returns_existing_uv(self) -> None:
        with patch("scripts._common.find_uv", return_value="/usr/bin/uv"):
            self.assertEqual(common.require_uv(), "/usr/bin/uv")

    def test_require_uv_raises_when_missing(self) -> None:
        with patch("scripts._common.find_uv", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                common.require_uv()
        self.assertIn("uv is not installed", str(ctx.exception))


class NvidiaSmiTests(unittest.TestCase):
    def test_has_nvidia_smi_true_when_command_succeeds(self) -> None:
        with patch("scripts._common.platform.system", return_value="Linux"):
            with patch(
                "scripts._common.subprocess.run",
                return_value=CompletedProcess(args=["nvidia-smi", "-L"], returncode=0),
            ):
                self.assertTrue(common.has_nvidia_smi())

    def test_has_nvidia_smi_false_on_command_error(self) -> None:
        with patch("scripts._common.platform.system", return_value="Windows"):
            with patch("scripts._common.subprocess.run", side_effect=OSError):
                self.assertFalse(common.has_nvidia_smi())

    def test_has_nvidia_smi_false_on_unsupported_platform(self) -> None:
        with patch("scripts._common.platform.system", return_value="Darwin"):
            with patch("scripts._common.subprocess.run") as mocked_run:
                self.assertFalse(common.has_nvidia_smi())
            mocked_run.assert_not_called()


class DetectTorchProfileTests(unittest.TestCase):
    def test_detect_torch_profile_cuda_on_windows_with_nvidia(self) -> None:
        with patch("scripts.install.platform.system", return_value="Windows"):
            with patch("scripts.install.has_nvidia_smi", return_value=True):
                self.assertEqual(install.detect_torch_profile(), "cuda_nightly")

    def test_detect_torch_profile_cpu_on_linux_without_nvidia(self) -> None:
        with patch("scripts.install.platform.system", return_value="Linux"):
            with patch("scripts.install.has_nvidia_smi", return_value=False):
                self.assertEqual(install.detect_torch_profile(), "cpu_stable")

    def test_detect_torch_profile_cpu_on_non_target_platform(self) -> None:
        with patch("scripts.install.platform.system", return_value="Darwin"):
            with patch("scripts.install.has_nvidia_smi", return_value=True):
                self.assertEqual(install.detect_torch_profile(), "cpu_stable")


class InstallFlowTests(unittest.TestCase):
    def test_install_applies_patch_before_first_verify(self) -> None:
        python_exec = Path("C:/virtual-project/.venv/Scripts/python.exe")
        events: list[str] = []

        def verify_stub(_python_exec: Path, _profile: str) -> bool:
            events.append("verify")
            return True

        with patch("scripts.install.require_uv", return_value="uv"):
            with patch("scripts.install.ensure_venv", return_value=python_exec):
                with patch("scripts.install.sync_project"):
                    with patch("scripts.install.detect_torch_profile", return_value="cpu_stable"):
                        with patch("scripts.install.install_torch_cpu_stable"):
                            with patch("scripts.install.pin_numpy_compat"):
                                with patch(
                                    "scripts.install.apply_patch",
                                    side_effect=lambda _python_exec: events.append("patch"),
                                ):
                                    with patch("scripts.install.verify_torch_stack", side_effect=verify_stub):
                                        result = install.main()

        self.assertEqual(result, 0)
        self.assertLess(events.index("patch"), events.index("verify"))

    def test_install_reapplies_patch_before_fallback_verify(self) -> None:
        python_exec = Path("C:/virtual-project/.venv/Scripts/python.exe")
        events: list[str] = []
        verify_outcomes = iter([False, True])

        def verify_stub(_python_exec: Path, profile: str) -> bool:
            events.append(f"verify:{profile}")
            return next(verify_outcomes)

        with patch("scripts.install.require_uv", return_value="uv"):
            with patch("scripts.install.ensure_venv", return_value=python_exec):
                with patch("scripts.install.sync_project"):
                    with patch("scripts.install.detect_torch_profile", return_value="cuda_nightly"):
                        with patch("scripts.install.install_torch_cuda_nightly"):
                            with patch("scripts.install.install_torch_cpu_stable"):
                                with patch("scripts.install.pin_numpy_compat"):
                                    with patch(
                                        "scripts.install.apply_patch",
                                        side_effect=lambda _python_exec: events.append("patch"),
                                    ):
                                        with patch("scripts.install.verify_torch_stack", side_effect=verify_stub):
                                            result = install.main()

        self.assertEqual(result, 0)
        first_patch_index = events.index("patch")
        second_patch_index = events.index("patch", first_patch_index + 1)
        first_verify_index = events.index("verify:cuda_nightly")
        second_verify_index = events.index("verify:cpu_stable")
        self.assertLess(first_patch_index, first_verify_index)
        self.assertLess(second_patch_index, second_verify_index)


class StartFlowTests(unittest.TestCase):
    def test_start_applies_patch_before_preflight(self) -> None:
        events: list[str] = []

        def run_stub(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
            _ = cwd
            _ = env
            if len(cmd) > 1 and Path(cmd[1]) == start.PATCH_SCRIPT:
                events.append("patch")
            elif len(cmd) > 1 and Path(cmd[1]) == start.APP_FILE:
                events.append("app")

        def preflight_stub(_python_exec: Path) -> bool:
            events.append("preflight")
            return True

        with patch("scripts.start.venv_python_path", return_value=Path(".")):
            with patch("scripts.start.run", side_effect=run_stub):
                with patch("scripts.start.preflight_runtime_imports", side_effect=preflight_stub):
                    with patch("scripts.start.detect_runtime_device", return_value="cpu"):
                        with patch.dict("os.environ", {}, clear=True):
                            result = start.main()

        self.assertEqual(result, 0)
        self.assertLess(events.index("patch"), events.index("preflight"))

    def test_start_honors_cpu_override(self) -> None:
        with patch("scripts.start.venv_python_path", return_value=Path(".")):
            with patch("scripts.start.run") as mocked_run:
                with patch("scripts.start.preflight_runtime_imports", return_value=True):
                    with patch("scripts.start.detect_runtime_device", return_value="cuda"):
                        with patch.dict("os.environ", {"QWEN_DEVICE": "cpu"}, clear=True):
                            result = start.main()

        self.assertEqual(result, 0)
        app_calls = [call for call in mocked_run.call_args_list if Path(call.args[0][1]) == start.APP_FILE]
        self.assertEqual(len(app_calls), 1)
        app_env = app_calls[0].kwargs["env"]
        self.assertEqual(app_env["QWEN_DEVICE"], "cpu")

    def test_start_fails_on_cuda_override_without_cuda(self) -> None:
        with patch("scripts.start.venv_python_path", return_value=Path(".")):
            with patch("scripts.start.run") as mocked_run:
                with patch("scripts.start.preflight_runtime_imports", return_value=True):
                    with patch("scripts.start.detect_runtime_device", return_value="cpu"):
                        with patch.dict("os.environ", {"QWEN_DEVICE": "cuda"}, clear=True):
                            result = start.main()

        self.assertEqual(result, 1)
        app_calls = [call for call in mocked_run.call_args_list if Path(call.args[0][1]) == start.APP_FILE]
        self.assertEqual(len(app_calls), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
