# 🏭 Qwen3 Voice Factory (RTX 50 Series Optimized)

## Overview

A local, portable GUI for [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice), focused on fast testing without complex node graphs (ComfyUI).
Specially optimized for **NVIDIA RTX 50 Series** (CUDA 12.8 / PyTorch Nightly), while still usable on previous generations (3090/4090) and CPU-only systems (slower).

![Screenshot](screenshot.png)

## Quick Start

1. Download this repository as a ZIP file and extract it.
2. Install dependencies and patch support:
   ```bash
   python scripts/install.py
   ```
3. Start the app:
   ```bash
   python scripts/start.py
   ```
4. The browser opens at `http://127.0.0.1:7860`.

What `scripts/install.py` does:
- Validates that **uv** is available on `PATH`.
- Creates a local `.venv` with Python 3.11.
- Auto-selects PyTorch profile:
  - CUDA Nightly for Windows/Linux systems with NVIDIA CUDA detected.
  - Stable CPU profile otherwise.
- Syncs dependencies from `pyproject.toml`.
- Applies the local 12Hz compatibility patch to `qwen_tts`.
- Verifies the runtime stack (`torch`, `torchvision`, `torchaudio`, `qwen_tts`) after patching.

## Features

- **🎬 Director:** Preset speakers with optional style/performance instructions.
- **🧬 Voice Cloner:** 3-10s reference audio cloning with optional transcript-guided high-quality mode.
- **🎨 Voice Creator:** Voice creation from text description + optional performance instruction.
- **📊 Live Hardware Monitor:** Real-time CPU/RAM/VRAM status in the UI.
- **📂 Auto-Save:** Every generation is saved in `outputs_audio/` with a timestamp.
- **Portable workflow:** Everything stays inside the project folder.

## Runtime Behavior

- On first use of a tab, the corresponding model is downloaded from Hugging Face (~4GB each).
- The 12Hz-only `qwen_tts` patch is applied before verification during install and re-applied on startup (idempotent).
- Runtime mode defaults to auto-detection at startup (`cuda` when available, otherwise `cpu`).
- You can override runtime mode with `QWEN_DEVICE=cpu` or `QWEN_DEVICE=cuda` when launching `scripts/start.py`.
- If `QWEN_DEVICE=cuda` is requested but CUDA is unavailable, startup exits with an explicit error.
- Patch details: [`docs/qwen-tts-patch.md`](docs/qwen-tts-patch.md).

## Models

- `Director`: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `Voice Cloner`: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `Voice Creator`: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`

## Requirements

- Windows 10/11 or Linux
- NVIDIA GPU (recommended: 12GB+ VRAM) for best performance
- CPU-only mode is supported but significantly slower
- Internet connection (required for install and model downloads)

## 🔗 Credits & Acknowledgements

This project is a GUI wrapper built to make the work of the **Qwen Team** easy to use locally.

- **Base models:** [Alibaba Cloud / Qwen Team](https://huggingface.co/Qwen)
- Please support the original work on Hugging Face and GitHub.
