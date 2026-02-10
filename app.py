import os
import sys
import warnings
import logging
from typing import Literal


import gradio as gr
from gradio.themes import Default
import torch
import soundfile as sf
import time
import librosa
import numpy as np
import psutil
from qwen_tts import Qwen3TTSModel

# --- CONFIGURATION ---
OUTPUT_DIR = "outputs_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ModelMode = Literal["director", "cloner", "designer"]

# Model Definitions
MODELS_CONFIG: dict[ModelMode, str] = {
    "director": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "cloner":   "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "designer": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
}

# Global Model Cache
loaded_models: dict[ModelMode, Qwen3TTSModel | None] = {
    "director": None,
    "cloner": None,
    "designer": None
}

RUNTIME_DEVICE = os.environ.get("QWEN_DEVICE", "auto").strip().lower()
if RUNTIME_DEVICE not in {"auto", "cuda", "cpu"}:
    RUNTIME_DEVICE = "auto"

# --- SYSTEM MONITOR ---
def get_system_stats():
    """System Monitor (CPU/RAM/VRAM) with HTML Styling."""
    try:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        
        vram_display = "<span class='vf-vram-na'>N/A</span>"
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            used_gb = used / (1024**3)
            total_gb = total / (1024**3)
            percent = (used / total) * 100
            vram_class = "vf-vram-warn" if percent > 90 else "vf-vram-ok"
            vram_display = f"<span class='{vram_class}'>{used_gb:.1f}GB</span> / {total_gb:.1f}GB ({percent:.0f}%)"
        
        return f"""
        <div class="vf-monitor">
            <div class="vf-monitor-card">
                🖥️ CPU: {cpu}%
            </div>
            <div class="vf-monitor-card">
                🧠 RAM: {ram}%
            </div>
            <div class="vf-monitor-card">
                🎮 VRAM: {vram_display}
            </div>
        </div>
        """
    except Exception:
        return "Loading Stats..."

# --- SMART MODEL LOADER ---
def load_specific_model(mode: ModelMode) -> Qwen3TTSModel:
    global loaded_models
    cached_model = loaded_models[mode]
    if cached_model is not None:
        return cached_model
    
    model_id = MODELS_CONFIG[mode]
    print(f"⏳ Loading model for '{mode}': {model_id} ...")

    if RUNTIME_DEVICE == "cuda":
        runtime_device = "cuda"
    elif RUNTIME_DEVICE == "cpu":
        runtime_device = "cpu"
    else:
        runtime_device = "cuda" if torch.cuda.is_available() else "cpu"

    if runtime_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime requested but was not detected.")

    if runtime_device == "cuda":
        # GPU profile tuned for RTX 50 series.
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map="cuda",
            dtype=torch.bfloat16,
            attn_implementation="sdpa"
        )
        mode_label = "CUDA/SDPA"
    else:
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map="cpu",
            dtype=torch.float32
        )
        mode_label = "CPU"

    print(f"✅ {mode.upper()} model loaded successfully ({mode_label} Mode)!")
    loaded_models[mode] = model
    return model

# --- ENGINE 1: DIRECTOR ---
def run_director(text, speaker, instruction):
    if not text or len(text.strip()) == 0: return None, "⚠️ Please enter text first!"
    
    try:
        model = load_specific_model("director")
    except Exception as e:
        return handle_error(e)
    
    print(f"🎬 Director: '{text}' | Speaker: {speaker}")
    try:
        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker=speaker, 
            instruct=instruction if instruction else None,
            language="Auto"
        )
        return save_audio(wavs, sr, f"director_{speaker}"), "Done"
    except Exception as e: return handle_error(e)

# --- ENGINE 2: CLONER ---
def run_cloner(text, ref_audio, ref_text):
    if not ref_audio: return None, "⚠️ No Audio provided!"
    if not text or len(text.strip()) == 0: return None, "⚠️ Please enter text first!"
    
    try:
        model = load_specific_model("cloner")
    except Exception as e:
        return handle_error(e)
    
    # 1. Load Audio
    try:
        ref_wav, ref_sr = librosa.load(ref_audio, sr=16000, mono=True)
        ref_sr = int(ref_sr)
    except Exception as e:
        return None, f"⚠️ Error loading audio file: {e}"

    # 2. CLONING PROCESS
    print(f"🧬 Cloning: '{text}'")
    try:
        if ref_text and len(ref_text.strip()) > 0:
            print(f"👉 High-Quality Mode (ICL) with Transcript")
            use_x_vector = False
            actual_ref_text = ref_text
        else:
            print("👉 Fast Mode (X-Vector)")
            use_x_vector = True
            actual_ref_text = None

        ref_audio_input: tuple[np.ndarray, int] = (ref_wav, ref_sr)
        wavs, sr = model.generate_voice_clone(
            text=text,
            ref_audio=ref_audio_input,
            ref_text=actual_ref_text,
            x_vector_only_mode=use_x_vector,
            language="Auto"
        )
        return save_audio(wavs, sr, "clone"), "Done"
    except Exception as e: return handle_error(e)

# --- ENGINE 3: CREATOR ---
def run_designer(text, voice_description, instruction):
    if not text or len(text.strip()) == 0: return None, "⚠️ Please enter text first!"
    if not voice_description or len(voice_description.strip()) == 0: return None, "⚠️ Please describe the voice first!"
    
    try:
        model = load_specific_model("designer")
    except Exception as e:
        return handle_error(e)
    
    print(f"🎨 Design: '{text}'")
    try:
        final_instruct = instruction
        if not final_instruct or len(final_instruct.strip()) == 0:
            final_instruct = voice_description

        wavs, sr = model.generate_voice_design(
            text=text,
            voice_description=voice_description, 
            instruct=final_instruct,
            language="Auto"
        )
        return save_audio(wavs, sr, "design"), "Done"
    except Exception as e: return handle_error(e)

# --- HELPERS ---
def save_audio(wavs, sr, prefix):
    timestamp = int(time.time())
    # Use absolute path
    filename = f"{prefix}_{timestamp}.wav"
    path = os.path.abspath(os.path.join(OUTPUT_DIR, filename))
    
    if isinstance(wavs, list):
        if len(wavs) == 0:
            raise ValueError("Model returned empty audio list.")
        data = wavs[0]
    else: data = wavs
    
    sf.write(path, data, sr)
    return path

def handle_error(e):
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    return None, f"Error: {e}"

# --- GUI SETUP ---
custom_css = """
.gen-btn {
    background: linear-gradient(90deg, #ff9966, #ff5e62);
    color: #fff;
    border: 1px solid transparent;
    font-weight: 700;
}

.gen-btn:hover {
    filter: brightness(1.04);
}

.header-row {
    align-items: center;
    margin-bottom: 20px;
    border-bottom: 1px solid var(--border-color-primary, #d1d5db);
    padding-bottom: 10px;
}

.vf-monitor {
    display: flex;
    gap: 12px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 14px;
    color: var(--body-text-color, #1f2937);
    align-items: center;
    justify-content: flex-end;
    height: 100%;
    flex-wrap: wrap;
}

.vf-monitor-card {
    background: var(--block-background-fill, #ffffff);
    border: 1px solid var(--border-color-primary, #d1d5db);
    border-radius: 6px;
    padding: 5px 10px;
    color: var(--body-text-color, #1f2937);
}

.vf-vram-ok {
    color: #0a8f45;
    font-weight: 600;
}

.dark .vf-vram-ok {
    color: #34d399;
}

.vf-vram-warn {
    color: #c62828;
    font-weight: 700;
}

.dark .vf-vram-warn {
    color: #f87171;
}

.vf-vram-na {
    color: var(--body-text-color-subdued, #6b7280);
}
"""
SPEAKERS = ["Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna", "Sohee"]


with gr.Blocks(title="Qwen3 Voice Factory") as demo:
    
    # --- HEADER ---
    with gr.Row(elem_classes="header-row"):
        with gr.Column(scale=1):
            gr.Markdown("# 🏭 Qwen3 Voice Factory")
            gr.Markdown("RTX 50 Series Powered | 12Hz Workflow | 3 Engines | Portable")
            gr.Markdown("Quick start: `python scripts/install.py` then `python scripts/start.py` | First use in each tab downloads its model (~4GB).")
        
        with gr.Column(scale=1):
            stats_display = gr.HTML(value=get_system_stats())
            timer = gr.Timer(1.0)
            timer.tick(get_system_stats, outputs=stats_display)

    # --- TABS ---
    with gr.Tabs():
        
        # TAB 1: DIRECTOR
        with gr.Tab("🎬 Director (Preset Speaker + Style)"):
            gr.Markdown("Use a preset speaker and optional style/performance instructions for directed delivery.")
            with gr.Row():
                with gr.Column():
                    t1_text = gr.Textbox(
                        label="Text to speak", 
                        placeholder="Example: Hello, I am using the director mode.", 
                        lines=2
                    )
                    with gr.Row():
                        t1_speaker = gr.Dropdown(SPEAKERS, value="Ryan", label="Speaker")
                        t1_instr = gr.Textbox(
                            label="Style Instruction (Optional)", 
                            placeholder="Optional: e.g. Angry, Whispering, Happy", 
                            lines=1
                        )
                    t1_btn = gr.Button("🔊 GENERATE", elem_classes="gen-btn")
                    t1_stat = gr.Textbox(label="Status / Error")
                with gr.Column():
                    t1_out = gr.Audio(label="Output")
            t1_btn.click(run_director, [t1_text, t1_speaker, t1_instr], [t1_out, t1_stat])

        # TAB 2: CLONER
        with gr.Tab("🧬 Voice Cloner (Transcript-Aware)"):
            gr.Markdown("Use 3-10s reference audio. Add an exact transcript for highest quality (ICL), or leave it empty for faster X-Vector mode.")
            with gr.Row():
                with gr.Column():
                    t2_text = gr.Textbox(
                        label="Text to speak", 
                        placeholder="Example: This is my cloned voice speaking.", 
                        lines=2
                    )
                    
                    # Simple Audio Input (Mic + Upload)
                    t2_ref = gr.Audio(
                        label="Reference Audio (3-10s)", 
                        type="filepath", 
                        sources=["microphone", "upload"]
                    )
                    gr.Markdown("*Tip: Provide the exact transcript for best quality. Generated outputs are auto-saved to `outputs_audio/`.*")
                    
                    t2_ref_text = gr.Textbox(
                        label="Transcript of Audio (Optional)", 
                        placeholder="Optional: Write exactly what is said in the audio for higher quality.", 
                        lines=1
                    )
                    t2_btn = gr.Button("🧬 CLONE VOICE", elem_classes="gen-btn")
                    t2_stat = gr.Textbox(label="Status / Error")
                with gr.Column():
                    t2_out = gr.Audio(label="Output")
            t2_btn.click(run_cloner, [t2_text, t2_ref, t2_ref_text], [t2_out, t2_stat])

        # TAB 3: CREATOR
        with gr.Tab("🎨 Voice Creator (Description + Style)"):
            gr.Markdown("Describe who the voice is, then optionally guide how it should perform.")
            with gr.Row():
                with gr.Column():
                    t3_text = gr.Textbox(
                        label="Text to speak", 
                        placeholder="Example: I was created from a text description.", 
                        lines=2
                    )
                    t3_desc = gr.Textbox(
                        label="Voice Description (Who?)", 
                        placeholder="Example: A wise old wizard with a deep, raspy voice", 
                        lines=1
                    )
                    t3_instr = gr.Textbox(
                        label="Style/Performance (How?)", 
                        placeholder="Optional: Speaking slowly, whispering, shouting", 
                        lines=1
                    )
                    t3_btn = gr.Button("🎨 CREATE VOICE", elem_classes="gen-btn")
                    t3_stat = gr.Textbox(label="Status / Error")
                with gr.Column():
                    t3_out = gr.Audio(label="Output")
            t3_btn.click(run_designer, [t3_text, t3_desc, t3_instr], [t3_out, t3_stat])

if __name__ == "__main__":
    demo.launch(inbrowser=True, css=custom_css, theme="ocean")
