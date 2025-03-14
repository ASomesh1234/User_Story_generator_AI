import os
import librosa
import soundfile as sf
from transformers import pipeline
from pydub.silence import split_on_silence
import ollama


from pydub import AudioSegment
from transformers import pipeline


print(librosa.__version__)
# Use a pipeline as a high-level helper
from transformers import pipeline


#i need to be very aware of the thibks and le tbe aware of things tht should be

import google.generativeai as genai
# Configure the API key
genai.configure(api_key="AIzaSyDzr2N7iTDxTYYBQtlIV8A33_XLPzewgGU")


#client = genai.Client(api_key="AIzaSyDzr2N7iTDxTYYBQtlIV8A33_XLPzewgGU")





def transcribe_large_audio(audio_path):

   # transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")

    audio = AudioSegment.from_file(audio_path)
    chunks = split_on_silence(audio, min_silence_len= 500, silence_thresh=-40)

    full_text = ""
    for i, chunk in enumerate(chunks):
        chunk_path = f"temp_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        
        result = transcriber(chunk_path, return_timestamps=True)  # Enable timestamps
        full_text += result["text"] + " "

    return full_text

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes speech from an audio file using a Hugging Face model.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        str: Transcribed text.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

   
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    #transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    generation_config={"language": "en", "task": "transcribe"}
        #)
    # Load and preprocess the audio
    audio, sr = librosa.load(audio_path, sr=16000)  # Convert to 16kHz
    temp_audio_path = "temp_converted.wav"
    sf.write(temp_audio_path, audio, sr)

    # Transcribe audio
    print("🔍 Transcribing audio...")
    result = transcriber(temp_audio_path)
   # result = transcriber(temp_audio_path, generate_kwargs={"language": "en"})
    #result = transcriber(audio_path, generate_kwargs={"task": "translate"})

    transcription = result["text"]

    os.remove(temp_audio_path)  # Cleanup temp file

    print("✅ Transcription completed!")
    print(result["text"])
    print(transcription)
    return transcription




transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")

def transcribe_audio1(audio_path):
    audio = AudioSegment.from_file(audio_path)
    
    chunk_length_ms = 30 * 1000  
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    
    transcript = []
    
    for idx, chunk in enumerate(chunks):
        chunk_path = f"temp_chunk_{idx}.wav"
        chunk.export(chunk_path, format="wav")
        result = transcriber(chunk_path)["text"]
        transcript.append(result)
        print(f"Chunk {idx + 1}: {result}")
        print("✅ Transcription completed!")
    
    return " ".join(transcript)

#summarizer = pipeline("summarization", model="google/flan-t5-large")
#summarizer = pipeline("summarization", model="meta-llama/Meta-Llama-3-8B")
summarizer = pipeline("summarization", model="google/flan-t5-large")





def summarize_text(text: str) -> str:
    """
    Summarizes the given text using llama3.2.
    """
    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)[0]["summary_text"]
    return summary

def save_to_file(text, filename="transcription.txt"):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"✅ Text saved to {filename}")  

def read_from_txt(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
        print(f"✅ Text read from {filename}")
    return text  

def summarize_with_ollama(text, model="llama3"):
    prompt = f"Summarize the following text:\n\n{text}"
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]  
def summarize_with_google(text):
    prompt = f"Summarize the following text:\n\n{text}"
    pipe = pipeline("text2text-generation", model="CohereForAI/aya-101")
    response = pipe(prompt, max_length=200, min_length=50, do_sample=False)[0]["summary_text"]
    #model = genai.GenerativeModel("gemini-pro")
    #response = model.generate_content(prompt)
    return response  



# Test
if __name__ == "__main__":
    #audio_path = "/home/somesh/Downloads/5 HABITS that CHANGED my LIFE in 1 WEEK (THESE LESSONS WILL CHANGE YOUR LIFE) STOIC PHILOSOPHY.mp3"
    #audio_path = "/home/somesh/Documents/User_Stories_generator/User_Story_generator_AI/backend/temp_chunk_5.wav"
  #  transcribe_audio1(audio_path)
  #  text = transcribe_audio(audio_path)
   # text = " ".join(transcribe_audio1(audio_path))
   # transcribed_text = transcribe_large_audio(audio_path)
   # save_to_file(transcribed_text, "transcription.txt")
  #  print(transcribed_text)
   # print("Summery text")
   # summerizer_text = summarize_text(transcribed_text)
   # save_to_file(summerizer_text, "summerizer.txt")
  #  print(summerizer_text)
    text = read_from_txt("transcription.txt")
    print("Summery text -with started")
    summerizer_text1 = summarize_with_google(text)
    print(summerizer_text1)

    #print(summarize_text(transcribed_text))
