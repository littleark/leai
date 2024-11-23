import streamlit as st
from gtts import gTTS
from io import BytesIO
import os
from tempfile import NamedTemporaryFile
from playsound import playsound

def play(text):
    """Play audio using playsound."""
    print('PLAYING', text)
    print('#############§')
    mp3_fp = BytesIO()
    tts = gTTS(text, lang='en', tld='co.uk')
    tts.write_to_fp(mp3_fp)

    # Write the BytesIO content to a temporary file
    mp3_fp.seek(0)  # Reset the pointer to the beginning of the BytesIO object
    with NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        tmp_file.write(mp3_fp.read())
        tmp_file_path = tmp_file.name  # Store the path to the temporary file

    # Play the mp3 file
    playsound(tmp_file_path)

    # Optionally, delete the temporary file after playing
    os.remove(tmp_file_path)

def initialize_tts():
    """Initialize TTS engine and stream."""
    # if 'tts_engine' not in st.session_state:
    #     st.session_state.tts_engine = CoquiEngine()
    #     # st.session_state.tts_engine = ElevenlabsEngine(elevenlabs_api_key)
    # if 'tts_stream' not in st.session_state:
    #     st.session_state.tts_stream = TextToAudioStream(st.session_state.tts_engine)


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
