import os

from pydub import AudioSegment
import ffmpeg

import torch
from transformers import pipeline
from langgraph.graph import StateGraph
import yt_dlp


# Define state class
class AudioState:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.audio_path = "extracted_audio.wav"
        self.transcription = ""


def extract_audio(state: AudioState):
    video = mp.VideoFileClip(state.video_path)
    video.audio.write_audiofile(state.audio_path)
    return state

# Function to transcribe audio using Hugging Face model
def transcribe_audio(state: AudioState):
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    state.transcription = transcriber(state.audio_path)["text"]
    return state


def download_youtube_video(youtube_url, output_path="downloaded_video.mp4"):
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
        print(f"Video downloaded to: {output_path}")
    return output_path

def extract_audio(video_path, output_audio_path="extracted_audio.wav"):
    audio = ffmpeg.input(video_path).output(output_audio_path).run(overwrite_output=True)
    return output_audio_path




# Define the LangGraph workflow
graph = StateGraph(AudioState)
graph.add_node("download_youtube_video", extract_audio)
graph.add_node("transcribe_audio", transcribe_audio)
graph.set_entry_point("extract_audio")
graph.add_edge("extract_audio", "transcribe_audio")

#workflow = graph.compile()

if __name__ == "__main__":
    print("🚀 Starting LangGraph workflow...")
    video_path = download_youtube_video("https://www.youtube.com/watch?v=9N6a-VLBa2I")
    extract_audio(video_path)
    print("🔍 Transcribing audio...")
      # Provide your video file path here
   # initial_state = AudioState(video_path)
   # final_state = workflow.invoke(initial_state)
    #print("Transcription:", final_state.transcription)
   # extract_audio(video_path)
