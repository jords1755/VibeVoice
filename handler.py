import os
import base64
import io
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import runpod

MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/VibeVoice-1.5B")
LANGUAGE = os.getenv("LANGUAGE", "en")
HF_TOKEN = os.getenv("HF_TOKEN", None)

print(f"Loading model: {MODEL_NAME} (lang={LANGUAGE})")
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_auth_token=HF_TOKEN
).to(device)

def synthesize(text: str, language: str):
    inputs = processor(text=text, language=language, return_tensors="pt").to(device)
    with torch.no_grad():
        audio_values = model.generate(**inputs)
    audio_values = audio_values.cpu()
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_values, 16000, format="wav")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def handler(event):
    text = event["input"].get("text")
    language = event["input"].get("language", LANGUAGE)
    if not text:
        return {"error": "Missing 'text' input"}
    audio_b64 = synthesize(text, language)
    return {"audio_base64": audio_b64}

runpod.serverless.start({"handler": handler})
