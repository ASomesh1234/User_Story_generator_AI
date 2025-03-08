from transformers import pipeline
import trascrpting  

import config


#summarizer = pipeline("summarization", model=config.SUMMARIZATION_MODEL)
summarizer = pipeline("summarization", model="google/flan-t5-large")


def summarize_text(text: str) -> str:
    """
    Summarizes the given text using google/flan-t5-large.
    """
    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)[0]["summary_text"]
    return summary


if __name__ == "__main__":
    summary = summarize_text(text)
    print(summary)
