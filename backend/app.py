import streamlit as st
import requests

st.title("User Story Generator")

input_text = st.text_area("Enter documentation text:")

if st.button("Generate User Stories"):
    response = requests.post("http://localhost:5000/generate_user_stories", json={"type": "text", "data": input_text})
    st.write(response.json())
