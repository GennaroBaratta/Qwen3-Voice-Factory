# 🏭 Qwen3 Voice Factory (RTX 50 Series Optimized)

A local, portable GUI for [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice).
Specially optimized for **NVIDIA RTX 50 Series** (CUDA 12.8 / PyTorch Nightly), but also runs on previous generations (3090/4090).

> **🎯 Perfect for anyone who wants to test these models quickly without dealing with complex node graphs (ComfyUI).**

![Screenshot](screenshot.png)

## Features

- **🎬 Director Mode:** Choose presets (Ryan, Vivian) and provide direction instructions ("Angry", "Whispering").
- **🧬 Voice Cloner:** Upload a short audio file (3-10s) and clone the voice (supports High-Quality ICL Mode).
- **🎨 Voice Creator:** Create completely new voices from scratch using text descriptions (Voice Design).
- **📊 Live Hardware Monitor:** Includes a real-time dashboard to watch your VRAM/RAM/CPU usage while generating.
- **📂 Auto-Save:** Automatically creates an `outputs_audio` folder and saves every generation with a timestamp.
- **Portable:** Does not modify your Windows system. Everything stays contained in one folder.

## Installation

1. Download this repository as a ZIP file and extract it.
2. Run:
   ```powershell
   python scripts/install.py
   ```
   - The script installs **uv** (Python manager) if needed.
   - It creates a local `.venv` with Python 3.11.
   - It installs PyTorch Nightly (required for Blackwell / RTX 50 Series support).
   - It syncs all project dependencies from `pyproject.toml`.
   - It applies a local compatibility patch (`scripts/patch_qwen_tts_v1_removal.py`) that keeps `qwen-tts` on the 12Hz tokenizer path only.
3. Wait until the installation is complete.

## Usage

1. Run:
   ```powershell
   python scripts/start.py
   ```
2. Your browser will open automatically at `http://127.0.0.1:7860`.

## Models
Models are automatically downloaded from HuggingFace the first time you use a specific tab (~4GB per model). Please ensure you have enough disk space.

## Requirements

- Windows 10/11
- NVIDIA GPU (Recommended: 12GB+ VRAM)
- Internet connection (required for installation and model download)

## 🔗 Credits & Acknowledgements

This project is a GUI wrapper built to make the amazing work of the **Qwen Team** easily accessible. All AI capabilities are powered by their models.

- **Base Models:** Developed by [Alibaba Cloud / Qwen Team](https://huggingface.co/Qwen).
- Please support their original work on HuggingFace and GitHub.

## 🤝 Support

This is a free open-source project. I don't ask for donations.
However, if you want to say "Thanks", check out my profile on **[Spotify](https://open.spotify.com/artist/7EdK2cuIo7xTAacutHs9gv?si=4AqQE6GcQpKJFeVk6gJ06g)**.

A follow or a listen is the best way to support me! 🎧
