from transcription import transcribe_audio
from retrieval import retrieve_docs
from summarization import summarize_text
from story_generation import generate_user_stories
from state_management import UserStoryState
from langgraph.graph import Graph

# ✅ Define LangGraph Workflow
graph = Graph(UserStoryState)

# Step 1: Transcribe Audio/Video
@graph.add_node
def transcribe(state):
    if state.input_type in ["audio", "video"]:
        state.transcribed_text = transcribe_audio(state.input_data)
    return state

# Step 2: Retrieve Documentation
@graph.add_node
def retrieve(state):
    query = state.transcribed_text if state.transcribed_text else state.input_data
    state.retrieved_docs = retrieve_docs(query)
    return state

# Step 3: Summarize Retrieved Documentation
@graph.add_node
def summarize(state):
    state.summary = summarize_text(" ".join(state.retrieved_docs))
    return state

# Step 4: Generate User Stories
@graph.add_node
def generate(state):
    state.user_stories = generate_user_stories(state.summary)
    return state

# ✅ Define Graph Execution Order
graph.set_entry_node(transcribe)
graph.add_edge(transcribe, retrieve)
graph.add_edge(retrieve, summarize)
graph.add_edge(summarize, generate)

# ✅ Compile Graph
workflow = graph.compile()
