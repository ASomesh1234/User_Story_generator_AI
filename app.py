# frontend/app.py
import streamlit as st
from workflows.graph import workflow
from workflows.state import AgentState

st.title("📜 LangGraph Project Analyzer")

option = st.radio("Choose an input type:", ("Upload Video", "Upload Document"))

if option == "Upload Video":
    video_file = st.file_uploader("Upload a video", type=["mp4", "wav"])
    if video_file:
        state = AgentState(video_path=video_file.name)
        result = workflow.invoke(state)
        st.write("Transcript:", result.transcript)
        st.write("User Stories:", result.user_stories)

elif option == "Upload Document":
    doc_file = st.file_uploader("Upload a document", type=["txt", "pdf"])
    if doc_file:
        document_path = doc_file.name  # Store document path
        document_text = doc_file.read().decode("utf-8")
        state = AgentState(document_path=document_path, document_text=document_text)
        result = workflow.invoke(state)
        st.write("Project Requirements:", result.project_requirements)
