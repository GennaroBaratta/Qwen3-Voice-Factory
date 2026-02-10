from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


CORE_INIT_OLD = """from .tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Config
from .tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Model
from .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
"""

CORE_INIT_NEW = """from .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
"""

TOKENIZER_IMPORT_OLD = """from ..core import (
    Qwen3TTSTokenizerV1Config,
    Qwen3TTSTokenizerV1Model,
    Qwen3TTSTokenizerV2Config,
    Qwen3TTSTokenizerV2Model,
)
"""

TOKENIZER_IMPORT_NEW = """from ..core import (
    Qwen3TTSTokenizerV2Config,
    Qwen3TTSTokenizerV2Model,
)
"""

TOKENIZER_REGISTER_OLD = """        AutoConfig.register("qwen3_tts_tokenizer_25hz", Qwen3TTSTokenizerV1Config)
        AutoModel.register(Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV1Model)

        AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
        AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)
"""

TOKENIZER_REGISTER_NEW = """        AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
        AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)
"""


def _replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    if old not in text:
        raise RuntimeError(f"Cannot patch {label}: expected upstream block not found.")
    return text.replace(old, new, 1), True


def _strip_tokenizer_v1_imports(text: str) -> tuple[str, bool]:
    patterns = [
        r"^from \.tokenizer_25hz\.configuration_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Config\s*\n",
        r"^from \.tokenizer_25hz\.modeling_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Model\s*\n",
    ]
    changed = False
    for pattern in patterns:
        text_new, count = re.subn(pattern, "", text, count=1, flags=re.MULTILINE)
        if count:
            changed = True
        text = text_new
    return text, changed


def _write_if_changed(path: Path, old: str, new: str, label: str) -> bool:
    content = path.read_text(encoding="utf-8")
    try:
        updated, changed = _replace_once(content, old, new, label)
    except RuntimeError:
        if label == "core/__init__.py imports":
            updated, changed = _strip_tokenizer_v1_imports(content)
            if not changed:
                raise
        else:
            raise
    if changed:
        path.write_text(updated, encoding="utf-8")
    return changed


def _get_qwen_tts_root() -> Path:
    spec = importlib.util.find_spec("qwen_tts")
    if spec is None or spec.origin is None:
        raise RuntimeError("qwen_tts is not installed in the current Python environment.")
    return Path(spec.origin).resolve().parent


def main() -> int:
    root = _get_qwen_tts_root()
    core_init = root / "core" / "__init__.py"
    tokenizer_file = root / "inference" / "qwen3_tts_tokenizer.py"

    changed = []
    if _write_if_changed(core_init, CORE_INIT_OLD, CORE_INIT_NEW, "core/__init__.py imports"):
        changed.append(str(core_init))

    content = tokenizer_file.read_text(encoding="utf-8")
    content, changed_imports = _replace_once(
        content, TOKENIZER_IMPORT_OLD, TOKENIZER_IMPORT_NEW, "qwen3_tts_tokenizer.py import block"
    )
    content, changed_register = _replace_once(
        content, TOKENIZER_REGISTER_OLD, TOKENIZER_REGISTER_NEW, "qwen3_tts_tokenizer.py register block"
    )
    if changed_imports or changed_register:
        tokenizer_file.write_text(content, encoding="utf-8")
        changed.append(str(tokenizer_file))

    if changed:
        print("Patched qwen_tts (12Hz-only tokenizer strategy):")
        for item in changed:
            print(f" - {item}")
    else:
        print("qwen_tts patch already applied (no changes).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Patch failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
