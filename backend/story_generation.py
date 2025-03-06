from transformers import pipeline
import config

# ✅ Load Llama 3.2 model for text generation
story_generator = pipeline("text-generation", model=config.SUMMARIZATION_MODEL)

def generate_user_stories(summary: str) -> list:
    """
    Generates multiple user stories from the summary using Llama 3.2.
    """
    prompt = f"Create multiple user stories based on the following documentation summary:\n{summary}"
    stories = story_generator(prompt, max_length=300, num_return_sequences=3)
    return [story["generated_text"] for story in stories]
