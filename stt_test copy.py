import streamlit as st
import threading
import asyncio

# Initialize shared state for transcription
shared_transcription = {"text": "", "listening": False}

# Initialize the DeepgramTranscriber
from deepgram_stt import DeepgramTranscriber
transcriber = DeepgramTranscriber()

def background_listen():
    """Background task for listening and transcribing."""
    print("background listen")
    loop = asyncio.new_event_loop()  # Create a new event loop for the thread
    asyncio.set_event_loop(loop)    # Set the new event loop
    result = loop.run_until_complete(transcriber.listen_and_transcribe())
    shared_transcription["text"] += f"\n{result}"
    shared_transcription["listening"] = False

# Start listening
if st.button("Start Listening"):
    print("start listening")
    if not shared_transcription["listening"]:
        shared_transcription["listening"] = True
        threading.Thread(target=background_listen, daemon=True).start()

# Stop listening
if st.button("Stop Listening"):
    print("stop listening")
    if shared_transcription["listening"]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(transcriber.stop_listening())
        shared_transcription["listening"] = False
        print("stopped listening")

# Display transcription
st.write("Transcription:")
st.text_area("Captured Text", shared_transcription["text"], height=300)
