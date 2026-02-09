import os
import sys
import warnings
import logging

# --- SILENCER BLOCK ---
# 1. Suppress Python Warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*flash-attn.*")
warnings.filterwarnings("ignore", message=".*SoX.*")

# 2. Mute Loggers (show errors only)
logging.getLogger("torchaudio").setLevel(logging.ERROR)
logging.getLogger("qwen_tts").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
# ----------------------------------------------

import gradio as gr
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

# Model Definitions
MODELS_CONFIG = {
    "director": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "cloner":   "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "designer": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
}

# Global Model Cache
loaded_models = {
    "director": None,
    "cloner": None,
    "designer": None
}

# --- SYSTEM MONITOR ---
def get_system_stats():
    """System Monitor (CPU/RAM/VRAM) with HTML Styling."""
    try:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        
        vram_display = "N/A"
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            used_gb = used / (1024**3)
            total_gb = total / (1024**3)
            percent = (used / total) * 100
            # Change color if VRAM usage is high (>90%)
            color = "#ff4444" if percent > 90 else "#00ff88"
            vram_display = f"<span style='color:{color}'>{used_gb:.1f}GB</span> / {total_gb:.1f}GB ({percent:.0f}%)"
        
        return f"""
        <div style="display: flex; gap: 20px; font-family: 'Consolas', monospace; font-size: 14px; color: #ccc; align-items: center; justify-content: flex-end; height: 100%;">
            <div style="background: #1a1f2e; padding: 5px 10px; border-radius: 6px; border: 1px solid #333;">
                🖥️ CPU: {cpu}%
            </div>
            <div style="background: #1a1f2e; padding: 5px 10px; border-radius: 6px; border: 1px solid #333;">
                🧠 RAM: {ram}%
            </div>
            <div style="background: #1a1f2e; padding: 5px 10px; border-radius: 6px; border: 1px solid #333;">
                🎮 VRAM: {vram_display}
            </div>
        </div>
        """
    except Exception:
        return "Loading Stats..."

# --- SMART MODEL LOADER ---
def load_specific_model(mode):
    global loaded_models
    if loaded_models[mode] is not None: return loaded_models[mode]
    
    model_id = MODELS_CONFIG[mode]
    print(f"⏳ Loading model for '{mode}': {model_id} ...")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but was not detected.")
    # Loading with bfloat16 and SDPA for RTX 50 series
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda",
        dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )
    print(f"✅ {mode.upper()} model loaded successfully (SDPA Mode)!")
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

        wavs, sr = model.generate_voice_clone(
            text=text,
            ref_audio=(ref_wav, ref_sr),
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
body { background-color: #0b0f19; color: #fff; } 
gradio-app { background: #0b0f19 !important; }
.gen-btn { background: linear-gradient(90deg, #ff9966, #ff5e62); color: white; border: none; font-weight: bold; }
.header-row { align-items: center; margin-bottom: 20px; border-bottom: 1px solid #333; padding-bottom: 10px; }
"""
SPEAKERS = ["Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna", "Sohee"]

spotify_html = """
<div style="text-align: center; margin-top: 10px;">
    If you find this tool helpful, support me on 
    <a href="https://open.spotify.com/artist/7EdK2cuIo7xTAacutHs9gv?si=5d3AbCKgR3GemCemctb8FA" target="_blank" style="color: #1DB954; font-weight: bold; text-decoration: none;">Spotify</a>.
</div>
"""

with gr.Blocks(title="Qwen3 Voice Factory") as demo:
    
    # --- HEADER ---
    with gr.Row(elem_classes="header-row"):
        with gr.Column(scale=1):
            gr.Markdown("# 🏭 Qwen3 Voice Factory")
            # --- CHANGED: More professional subtitle ---
            gr.Markdown("RTX 50 Series Powered | 3 Engines | Portable")
        
        with gr.Column(scale=1):
            stats_display = gr.HTML(value=get_system_stats())
            timer = gr.Timer(1.0)
            timer.tick(get_system_stats, outputs=stats_display)

    # --- TABS ---
    with gr.Tabs():
        
        # TAB 1: DIRECTOR
        with gr.Tab("🎬 Director (Presets)"):
            with gr.Row():
                with gr.Column():
                    t1_text = gr.Textbox(
                        label="Text", 
                        placeholder="Example: Hello, I am using the director mode.", 
                        lines=2
                    )
                    with gr.Row():
                        t1_speaker = gr.Dropdown(SPEAKERS, value="Ryan", label="Speaker")
                        t1_instr = gr.Textbox(
                            label="Style/Instruction", 
                            placeholder="Optional: e.g. Angry, Whispering, Happy", 
                            lines=1
                        )
                    t1_btn = gr.Button("🔊 GENERATE", elem_classes="gen-btn")
                    t1_stat = gr.Textbox(label="Status")
                with gr.Column():
                    t1_out = gr.Audio(label="Output")
                    gr.Markdown(spotify_html)
            t1_btn.click(run_director, [t1_text, t1_speaker, t1_instr], [t1_out, t1_stat])

        # TAB 2: CLONER
        with gr.Tab("🧬 Voice Cloner"):
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
                    # Hint for User
                    gr.Markdown("*Note: To save your recording, use the download button in the audio player after recording.*")
                    
                    t2_ref_text = gr.Textbox(
                        label="Transcript of Audio", 
                        placeholder="Optional: Write exactly what is said in the audio for higher quality.", 
                        lines=1
                    )
                    t2_btn = gr.Button("🧬 CLONE VOICE", elem_classes="gen-btn")
                    t2_stat = gr.Textbox(label="Status")
                with gr.Column():
                    t2_out = gr.Audio(label="Output")
                    gr.Markdown(spotify_html)
            t2_btn.click(run_cloner, [t2_text, t2_ref, t2_ref_text], [t2_out, t2_stat])

        # TAB 3: CREATOR
        with gr.Tab("🎨 Voice Creator"):
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
                    t3_stat = gr.Textbox(label="Status")
                with gr.Column():
                    t3_out = gr.Audio(label="Output")
                    gr.Markdown(spotify_html)
            t3_btn.click(run_designer, [t3_text, t3_desc, t3_instr], [t3_out, t3_stat])

if __name__ == "__main__":
    demo.launch(inbrowser=True, css=custom_css)
