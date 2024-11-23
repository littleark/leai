# tts_module.py
import os
from dotenv import load_dotenv
from RealtimeTTS import TextToAudioStream, CoquiEngine, ElevenlabsEngine

load_dotenv()
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

# TTS State
tts_state = {
    "tts_engine": None,
    "tts_stream": None,
    "tts_enabled": False
}

def initialize_tts():
    """Initialize TTS engine and stream."""
    if tts_state["tts_engine"] is None:
        tts_state["tts_engine"] = CoquiEngine()
    if tts_state["tts_stream"] is None:
        tts_state["tts_stream"] = TextToAudioStream(tts_state["tts_engine"])

def play_audio(text):
    """Play audio directly."""
    if tts_state["tts_enabled"]:
        try:
            if tts_state["tts_engine"] is None:
                initialize_tts()

            # Retrieve TTS stream
            tts_stream = tts_state["tts_stream"]

            # Play the text
            tts_stream.feed(text)
            tts_stream.play()
            print("Playing audio...")

        except Exception as e:
            print(f"TTS Error: {str(e)}")
