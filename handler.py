import os
import io
import base64
import torch
import torchaudio
import runpod

# Import VibeVoice inference utilities
from vibevoice.inference import load_model, tts_generate  # adjust to actual module paths

MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/VibeVoice-1.5B")
DEFAULT_LANGUAGE = os.getenv("LANGUAGE", "en")
HF_TOKEN = os.getenv("HF_TOKEN", None)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[VibeVoice] Loading model '{MODEL_NAME}' on {device}...")
model, processor = load_model(MODEL_NAME, HF_TOKEN, device)

def synthesize(text: str, language: str) -> str:
    """Generate speech audio from text and return as base64 WAV."""
    audio_tensor = tts_generate(model, processor, text, language, device)
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_tensor.cpu(), 16000, format="wav")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def handler(event):
    try:
        text = event["input"].get("text")
        language = event["input"].get("language", DEFAULT_LANGUAGE)
        if not text:
            return {"error": "Missing 'text' input"}
        audio_b64 = synthesize(text, language)
        return {"language": language, "audio_base64": audio_b64}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
