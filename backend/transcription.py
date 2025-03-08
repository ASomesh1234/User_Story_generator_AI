from transformers import pipeline
from milvus_db import insert_text
import config

file_path = "path_to_audio_file"
transcriber = pipeline("automatic-speech-recognition", model=config.TRANSCRIPTION_MODEL)

def transcribe_audio(file_path: str) -> str:
    """
    Converts audio/video to text using Whisper, then stores it in Milvus.
    """
    transcript = transcriber(file_path)["text"]
    insert_text(transcript)  
    return transcript
