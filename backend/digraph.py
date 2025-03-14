from typing import TypedDict
from langgraph.graph import StateGraph, START

import os
import librosa
import soundfile as sf
from transformers import pipeline
from pydub.silence import split_on_silence

from pydub import AudioSegment
from langchain_groq import ChatGroq 
from groq import Groq
import streamlit as st
from graphviz import Digraph
import docx
import pdfplumber
from graphviz import Source

client = Groq(api_key="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")


class UserStoryGenerator(TypedDict):
    input_file: str
    extract_audio_from_video: str
    extract_text_from_docx:str
    extract_text: str
    summary: str
    project_requirement_document: str
    user_stories: str

def identify_file(state: UserStoryGenerator):
    file_path = state["input_file"]
    if file_path.endswith('mp3') or file_path.endswith('wav'):
        print("i am audio file in the identify_file ")
        return "extract_text_node"
       # return {"identify_file":file_path}
    elif file_path.endswith('mp4'):
        print("i am video file")
        return "extract_audio_from_video_node"
    elif file_path.endswith('pdf') or file_path.endswith('docx'):
        print(" I am document")
        return "extract_text_from_docx_node"
    else:
        print("please upload the valid Document")
        return f'please upload the valid Document'


def extract_text_from_docx(state:UserStoryGenerator) :
    print("i am document in the extract_text_from_docx")
    file_path = state["input_file"]
    ful_text = ""
    if file_path.endswith('docx'):
       doc = docx.Document(file_path)
       print("i am document in the extract_text_from_docx in the docx")
       text = "\n".join([para.text for para in doc.paragraphs])
       ful_text = text.strip()
       return {"extract_text_from_docx_node": ful_text}
    else:
         text = ""   
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        ful_text = text.strip()
        return {"extract_text_from_docx_node": ful_text}
     


def extract_audio_from_video(state:UserStoryGenerator):
    print("i am a video in the extract_audio_from_video ")
    file_path = state["input_file"]
    
    audio = AudioSegment.from_file(file_path)
    audio_file = audio.export("output_audio.wav", format="wav")
    return {"extract_audio_from_video_node": audio_file}

def extract_text(state: UserStoryGenerator):
    print("I am on the extract_text")
   # file_path = state.get("extract_audio_from_video", "input_file")
    full_text = ""
    file = state["input_file"]

    #if file_path.endswith('mp4'):
      # full_text = "i am video"
        
   # else:
    if "extract_audio_from_video" in state:
       print("i am extract_audio_from_video-")
       file_path = state["extract_audio_from_video"]
       full_text=  textfromaudio(file_path)
       return full_text
       
    else:
       file_path = file
       print("i am input state file in the extract_audio_from_video")
   
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
def textfromaudio(file_path):
    full_text =""
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")

    audio = AudioSegment.from_file(file_path)
    chunks = split_on_silence(audio, min_silence_len= 500, silence_thresh=-40)

        
    for i, chunk in enumerate(chunks):
        chunk_path = f"temp_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        print("in the video")
        
        result = transcriber(chunk_path, return_timestamps=True) 
        full_text += result["text"] + " "
    return  full_text   

def summarize_text(state: UserStoryGenerator):
    file = state["extract_text"]
    print("i am sumerixer text")
    extract = ""
    if "extract_text_from_docx" in state:

        extract = state["extract_text_from_docx"]
        return extract
    else:
        extract =file
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
    print("i am generate_prd")
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
builder.add_node("identify_file",identify_file)
builder.add_node("extract_text_from_docx_node",extract_text_from_docx)
builder.add_node("extract_audio_from_video_node",extract_audio_from_video)
builder.add_node("extract_text_node", extract_text)
builder.add_node("summarize_text", summarize_text)
builder.add_node("generate_prd", generate_prd)
builder.add_node("generate_user_stories", generate_user_stories)

#builder.add_edge(START, "extract_text_node")

builder.add_conditional_edges(START,identify_file )
builder.add_edge("extract_audio_from_video_node","extract_text_node")
builder.add_edge("extract_text_from_docx_node","summarize_text")
builder.add_edge("summarize_text","generate_prd")
builder.add_edge("extract_text_node", "summarize_text")
builder.add_edge("summarize_text", "generate_prd")
builder.add_edge("generate_prd", "generate_user_stories")


#graph = builder.build()
graph = builder.compile()


 # Compile LangGraph

        
#graph = builder.get_graph()
print(graph.nodes)  # Prints all nodes
#print(builder.graph_repr())

teststrip=graph.get_graph().print_ascii()
print(teststrip)

dot = Digraph(format="png")

dot.node("Start", "START")
dot.node("identify_file", "Identify File")
dot.node("extract_audio_from_video_node", "Extract Audio from Video")
dot.node("extract_text_from_docx_node", "Extract Text from DOCX")
dot.node("extract_text_node", "Extract Text")
dot.node("summarize_text", "Summarize Text")
dot.node("generate_prd", "Generate PRD")
dot.node("generate_user_stories", "Generate User Stories")

# Add edges
dot.edge("Start", "identify_file")
dot.edge("extract_audio_from_video_node", "extract_text_node")
dot.edge("extract_text_from_docx_node", "summarize_text")
dot.edge("extract_text_node", "summarize_text")
dot.edge("summarize_text", "generate_prd")
dot.edge("generate_prd", "generate_user_stories")
















if __name__ == '__main__':

 # initial_input = {"input_file": "/home/somesh/Downloads/videoplayback.mp4"}
  #f#or event in graph.stream(initial_input, stream_mode="messages"):
   # print(event)
 # e=graph.invoke(initial_input)
 # print(e)

  import streamlit as st

st.title("User Story Generator 📜")

uploaded_file = st.file_uploader("Upload an audio file (MP3, WAV, MP4)", type=["mp3", "wav", "mp4"])

if uploaded_file:
    file_path = f"./temp_uploaded.{uploaded_file.name.split('.')[-1]}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully! Processing...")

    initial_input = {"input_file": file_path}
    output = graph.invoke(initial_input)

    # Get the ASCII representation of the graph
    teststrip = graph.get_graph().print_ascii()

    st.subheader("📜 Extracted Text:")
    st.write(output["extract_text"])

    st.subheader("🔍 Summary:")
    st.write(output["summary"])

    st.subheader("📄 Project Requirement Document (PRD):")
    st.write(output["project_requirement_document"])

    st.subheader("📝 User Stories:")
    st.write(output["user_stories"])

    