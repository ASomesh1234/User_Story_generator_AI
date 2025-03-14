import os
import ffmpeg

from transformers import pipeline
from groq import Groq
from tabulate import tabulate


#pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
client = Groq(api_key="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")
#transcriber = client.models.get("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
#transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")
#summery = client.models.get("summarization", "ollama")
#print(models)

class user_story_generator:
    def __init__(self, summary: str, model="llama3-8b-8192"):
        self.summary = summary
        self.model = model

    def generate(self):
        prompt = ("You are an expert Agile Product Owner. Based on the given summary, "
            "create multiple well-structured user stories following the format:\n"
            "- As a [user role], I want to [goal], so that [benefit].\n\n"f"Create unique user stories from the following summary:\n\n{self.summary}")
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content if response.choices else "No user stories generated."





def user_story_generator1(summary: str, model="llama3-8b-8192"):
    """
    Generate unique user stories from a given summary using a Groq model.

    :param summary: The text summary to generate user stories from.
    :param model: The name of the Groq model (default: "llama3-8b-8192").
    :return: A list of generated user stories or an error message.
    """

    try:
        print("🔍 Generating user stories...")

        # Refined prompt for better user stories
        prompt = (
            "You are an expert Agile Product Owner. Based on the given summary, "
            "create multiple well-structured user stories following the format:\n"
            "- As a [user role], I want to [goal], so that [benefit].\n\n"
            f"Summary:\n{summary}\n\nGenerate at least 3 user stories."
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Adjust for randomness
            max_tokens=500,   # Allow more tokens for multiple user stories
        )

        # Debugging response
        print("🔄 Raw Response:", response)

        # Extract and return user stories
        if response.choices:
            user_stories = response.choices[0].message.content.strip().split("\n")
            return [story.strip() for story in user_stories if story.strip()]
        else:
            return ["No user stories generated."]

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return [f"Error: {str(e)}"]


def summarize_with_groq(text, model="llama3-8b-8192"):
    """
    Summarizes the given text using a Groq model.

    :param text: The text to summarize
    :param model: The name of the Groq model (default: "llama3-8b-8192")
    :return: The summarized text
    """
    try:
        prompt = f"Summarize the following text:\n\n{text}"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Adjust for randomness (0.0 = deterministic)
            max_tokens=300,   # Adjust summary length
        )

        # Extract and return the summary
        return response.choices[0].message.content if response.choices else "No response received."
    
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
#text = "Groq provides a high-performance LLM API that enables developers to integrate AI models seamlessly into applications."



def read_from_txt(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
        print(f"✅ Text read from {filename}")
    return text 





class user_story_generator123:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.audio_path = "extracted_audio.wav"
        self.transcription = ""
        self.summary = ""
        self.user_story = ""
    

def extract_audio(state: user_story_generator):
    audio = ffmpeg.input(state.video_path).output(state.audio_path).run(overwrite_output=True)
    return state

def transcribe_audio(state: user_story_generator):
    pass

    


if __name__ == "__main__":
    #GROQ_API_KEY ="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"
 #   client = Groq(api_key="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")
#models = client.models.list(filter="type == 'automatic-speech-recognition'")
#print(tabulate(models, headers="keys", tablefmt="grid"))
#print(tabulate(models, headers="keys", tablefmt="grid"))
   print("🚀 Starting LangGraph workflow...")
   text = read_from_txt("transcription.txt")
   print("🔍 Summarizing text...")
   summary = summarize_with_groq(text)

   print(summary)
   print("✅ Summary completed!")
   print("🔍 Generating user stories...")
   
   user_story = user_story_generator(summary)
   user_story2 = user_story.generate()
  # story= user_story.generate(summary)
   print(user_story2)