import streamlit as st
from RealtimeTTS import TextToAudioStream, CoquiEngine, ElevenlabsEngine
# import threading
import os
from dotenv import load_dotenv
from time import sleep

load_dotenv()
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

def initialize_tts():
    """Initialize TTS engine and stream."""
    if 'tts_engine' not in st.session_state:
        st.session_state.tts_engine = CoquiEngine()
        # st.session_state.tts_engine = ElevenlabsEngine(elevenlabs_api_key)
    if 'tts_stream' not in st.session_state:
        st.session_state.tts_stream = TextToAudioStream(st.session_state.tts_engine)


def play_audio(text):
    """Play audio directly, avoiding threading for Streamlit compatibility."""
    if st.session_state.get("tts_enabled", False):
        try:
            if 'tts_engine' not in st.session_state:
                initialize_tts()

            # Retrieve TTS engine and stream
            tts_stream = st.session_state.tts_stream

            # Play the text
            tts_stream.feed(text)
            tts_stream.play()
        except Exception as e:
            st.error(f"TTS Error: {str(e)}")
