from langgraph.graph import State
from typing import Optional, List

class UserStoryState(State):
    """
    Defines state management for user story generation.
    """
    input_type: str 
    input_data: str 
    transcribed_text: Optional[str] = None 
    retrieved_docs: List[str] = []  
    summary: str = ""  
    user_stories: List[str] = [] 