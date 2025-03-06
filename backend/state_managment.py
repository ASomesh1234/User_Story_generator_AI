from langgraph.graph import State
from typing import Optional, List

class UserStoryState(State):
    """
    Defines state management for user story generation.
    """
    input_type: str  # "audio", "video", "text"
    input_data: str  # File path or text
    transcribed_text: Optional[str] = None  # Stores transcriptions
    retrieved_docs: List[str] = []  # Retrieved documentation
    summary: str = ""  # Summarized text
    user_stories: List[str] = []  # Generated user stories
