from transformers import pipeline
import config

# ✅ Load Llama 3.2 model for summarization
summarizer = pipeline("summarization", model=config.SUMMARIZATION_MODEL)

def summarize_text(text: str) -> str:
    """
    Summarizes the given text using Llama 3.2.
    """
    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)[0]["summary_text"]
    return summary
