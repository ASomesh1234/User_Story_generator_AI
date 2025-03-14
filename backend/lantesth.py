from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import Dict
from groq import Groq
from langchain_groq import ChatGroq 





client = Groq(api_key="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")
from langgraph.graph import StateGraph, END
from typing import Dict


class UserStoryState:
    def __init__(self, summary: str):
        self.summary = summary
        self.story = None  


def generate_user_story(state: UserStoryState) -> Dict:
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)

    prompt = f"Create unique user stories from the following summary:\n\n{state.summary}"

    response = llm.invoke(prompt)
    story_text = response.content if response else "No user stories generated."

    return {"summary": state.summary, "story": story_text}  # ✅ Correct return type



class UserStoryState1:
    def __init__(self, summary: str):
        self.summary = summary
        self.story = None 


def generate_user_story1(state: UserStoryState) -> Dict:
    #llm = ChatOpenAI(model="llama3-8b-8192", temperature=0.7)
    llm = client.models.get("text-generation", "llama3-8b-8192")
    
    prompt = f"Create unique user stories from the following summary:\n\n{state.summary}"
    
    response = llm.invoke(prompt)
    state.story = response.content if response else "No user stories generated."
    
    return {"story": state.story}  # Return updated state


workflow = StateGraph(UserStoryState)
workflow.add_node("generate", generate_user_story)
workflow.set_entry_point("generate")
workflow.add_edge("generate", END)  
graph = workflow.compile()


def main():
    summary = "A user wants to install Dish TV at their home and books an appointment."
    initial_state = UserStoryState(summary)
    result = graph.invoke(initial_state)
    
    print("Generated User Story:")
    print(result["story"])

if __name__ == "__main__":
    main()
