from typing import TypedDict
from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI
import os
import librosa
import soundfile as sf
from transformers import pipeline
from pydub.silence import split_on_silence
import ollama
from pydub import AudioSegment
from langchain_groq import ChatGroq 
from groq import Groq
import streamlit as st
from graphviz import Digraph

client = Groq(api_key="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")


class UserStoryGenerator(TypedDict):
    input_file: str
    extract_text: str
    summary: str
    project_requirement_document: str
    user_stories: str




def extract_text(state: UserStoryGenerator):
    file_path = state["input_file"]
    full_text = ""

    if file_path.endswith('mp4'):
       full_text = "i am video"
        
    else:
        transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")

        audio = AudioSegment.from_file(file_path)
        chunks = split_on_silence(audio, min_silence_len= 500, silence_thresh=-40)

        
    for i, chunk in enumerate(chunks):
        chunk_path = f"temp_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        print("in the video")
        
        result = transcriber(chunk_path, return_timestamps=True) 
        full_text += result["text"] + " "

     
       
    
    return {"extract_text": full_text}

def summarize_text(state: UserStoryGenerator):
   # llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)
    prompt = f"Summarize the following text concisely while preserving key details:\n\n{state['extract_text']}"
    response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
    print("*****",response.choices[0].message.content,"******")

    print("****",response)
    content=response.choices[0].message.content
    
    return {"summary": content}

def generate_prd(state: UserStoryGenerator):
    #llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)
    prompt = f"Based on the following text, generate a structured Project Requirement Document that includes the project overview, objectives, scope, functional and non-functional requirements, and any key constraints:\n\n{state['summary']}"
    response = client.chat.completions.create(
           # model="llama3-8b-8192",
            model = "llama-3.2-1b-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
    content = content=response.choices[0].message.content
    return {"project_requirement_document": content}


def generate_user_stories(state: UserStoryGenerator):
   # llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)
    prompt =f"create detailed user stories following the 'As a [user], I want [goal], so that [reason]' format. Ensure the user stories cover all key functionalities and use cases: in the list\n\n{state['project_requirement_document']}"
    response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
    content=response.choices[0].message.content
    
    return {"user_stories": content}


builder = StateGraph(UserStoryGenerator)
builder.add_node("extract_text_node", extract_text)
builder.add_node("summarize_text", summarize_text)
builder.add_node("generate_prd", generate_prd)
builder.add_node("generate_user_stories", generate_user_stories)

builder.add_edge(START, "extract_text_node")
builder.add_edge("extract_text_node", "summarize_text")
builder.add_edge("summarize_text", "generate_prd")
builder.add_edge("generate_prd", "generate_user_stories")


graph = builder.compile()



#def visualize_graph():
 #   dot = Digraph(format="png")
    
   
  #  dot.node("START", "START", shape="oval", color="green")
   # dot.node("extract_text_node", "Extract Text", shape="box")
   # dot.node("summarize_text", "Summarize Text", shape="box")
   # dot.node("generate_prd", "Generate PRD", shape="box")
   # dot.node("generate_user_stories", "Generate User Stories", shape="box")
    
   
    #dot.edge("START", "extract_text_node")
    #dot.edge("extract_text_node", "summarize_text")
    #dot.edge("summarize_text", "generate_prd")
    #dot.edge("generate_prd", "generate_user_stories")

    
    #dot.render("user_story_graph", view=True)  







if __name__ == '__main__':

  #initial_input = {"input_file": "/home/somesh/Documents/User_Stories_generator/extracted_audio.wav"}
  #f#or event in graph.stream(initial_input, stream_mode="messages"):
   # print(event)
 # e=graph.invoke(initial_input)
 # print(e)
  

  st.title("User Story Generator 📜")
  uploaded_file = st.file_uploader("Upload an audio file (MP3, WAV, MP4)", type=["mp3", "wav", "mp4"])

if uploaded_file:
    file_path = f"./temp_uploaded.{uploaded_file.name.split('.')[-1]}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully! Processing...")

    initial_input = {"input_file": file_path}
    output = graph.invoke(initial_input)

    st.subheader("📜 Extracted Text:")
    st.write(output["extract_text"])

    st.subheader("🔍 Summary:")
    st.write(output["summary"])

    st.subheader("📄 Project Requirement Document (PRD):")
    st.write(output["project_requirement_document"])

    st.subheader("📝 User Stories:")
    st.write(output["user_stories"])
