import os
import io
import base64
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import runpod

# Load environment variables from RunPod config
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/VibeVoice-1.5B")
DEFAULT_LANGUAGE = os.getenv("LANGUAGE", "en")
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor and model once at startup
print(f"[VibeVoice] Loading model '{MODEL_NAME}' on {device}...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
    use_auth_token=HF_TOKEN
).to(device)
model.eval()

def synthesize_speech(text: str, language: str) -> str:
    """Generate speech audio from text and return as base64 WAV."""
    inputs = processor(text=text, language=language, return_tensors="pt").to(device)
    with torch.no_grad():
        audio_values = model.generate(**inputs)
    audio_values = audio_values.cpu()

    # Save to in-memory WAV
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_values, 16000, format="wav")
    buffer.seek(0)

    # Encode to base64 for JSON transport
    return base64.b64encode(buffer.read()).decode("utf-8")

def handler(event):
    """RunPod Serverless handler."""
    try:
        text = event["input"].get("text")
        language = event["input"].get("language", DEFAULT_LANGUAGE)

        if not text:
            return {"error": "Missing required 'text' input."}

        audio_b64 = synthesize_speech(text, language)
        return {
            "language": language,
            "audio_base64": audio_b64
        }

    except Exception as e:
        return {"error": str(e)}

# Start the RunPod serverless loop
runpod.serverless.start({"handler": handler})
