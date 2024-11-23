# streamlit_app.py
import streamlit as st
import requests

st.title("Text-to-Speech Player")
text_input = st.text_input("Enter text to play as audio:")
tts_enabled = st.checkbox("Enable TTS", value=False)

if tts_enabled and st.button("Play Audio"):
    try:
        # Send the text to the TTS server
        response = requests.post("http://localhost:5001/speak", json={"text": text_input})
        if response.status_code == 200:
            st.success("Audio is playing.")
        else:
            st.error("TTS Server Error: " + response.json().get("message", "Unknown error"))
    except requests.exceptions.ConnectionError:
        st.error("Unable to connect to the TTS server.")
