from transformers import pipeline
from milvus_db import insert_text
import config

# ✅ Load Whisper model
transcriber = pipeline("automatic-speech-recognition", model=config.TRANSCRIPTION_MODEL)

def transcribe_audio(file_path: str) -> str:
    """
    Converts audio/video to text using Whisper, then stores it in Milvus.
    """
    transcript = transcriber(file_path)["text"]
    insert_text(transcript)  # Store transcript in Milvus
    return transcript
