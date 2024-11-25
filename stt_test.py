import streamlit as st
import asyncio
from deepgram_stt import DeepgramTranscriber

# Initialize the DeepgramTranscriber
transcriber = DeepgramTranscriber()

# Streamlit Session State
if "listening" not in st.session_state:
    st.session_state.listening = False
if "transcription" not in st.session_state:
    st.session_state.transcription = ""

# Synchronous wrapper for Streamlit
def listen_and_transcribe_sync():
    """Run the async function in a blocking way for Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(transcriber.listen_and_transcribe())

# Start listening and transcribing
if st.button("Start Listening"):
    if not st.session_state.listening:
        st.session_state.listening = True
        st.write("Listening...")
        transcription = listen_and_transcribe_sync()
        st.session_state.transcription += f"\n{transcription}"
        st.session_state.listening = False
        st.write("Done listening.")

# Stop listening
if st.button("Stop Listening"):
    if st.session_state.listening:
        st.session_state.listening = False
        st.session_state.transcription += "\nStopped listening."

# Display transcription
st.write("Transcription:")
st.text_area("Captured Text", st.session_state.transcription, height=300)
