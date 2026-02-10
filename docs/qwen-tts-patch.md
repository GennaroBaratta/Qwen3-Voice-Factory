# qwen-tts 12Hz-Only Patch

This repository applies a local patch to the installed `qwen_tts` package to follow a 12Hz-only tokenizer strategy (same direction as Qwen3-TTS PR #109).

## Why

- Avoid loading the legacy 25Hz tokenizer path during import.
- Prevent indirect SoX-related runtime checks/warnings in this project.
- Keep the patch version-controlled in this repository, instead of hand-editing `.venv` each time.

## What is patched

The script `scripts/patch_qwen_tts_v1_removal.py` updates two files in the active environment:

- `qwen_tts/core/__init__.py`
- `qwen_tts/inference/qwen3_tts_tokenizer.py`

Changes:

- Remove `Qwen3TTSTokenizerV1*` imports.
- Register only `qwen3_tts_tokenizer_12hz`.

## How it is applied

- Automatically in `scripts/install.py`.
- Re-checked on each `scripts/start.py` run (idempotent; no-op if already patched).

Manual run:

```powershell
.\.venv\Scripts\python.exe scripts\patch_qwen_tts_v1_removal.py
```

## Updating qwen-tts later

1. Update dependencies as usual.
2. Run the patch script.
3. If script fails with "expected upstream block not found", upstream changed:
   - inspect the two target files above,
   - adjust `scripts/patch_qwen_tts_v1_removal.py`,
   - commit the script update.
