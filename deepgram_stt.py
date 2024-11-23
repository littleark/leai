import os
import threading
import json
from dotenv import load_dotenv
from websockets.sync.client import connect
import pyaudio

load_dotenv()

# WebSocket Configuration
DEFAULT_URL = "wss://api.deepgram.com/v1/listen"
DEFAULT_TOKEN = os.getenv("DEEPGRAM_API_KEY", None)

# Audio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024  # Number of frames per buffer


class SpeechToText:
    def __init__(self, url=DEFAULT_URL, token=DEFAULT_TOKEN):
        self.url = url
        self.token = token
        self.transcription = []
        self.socket = None
        self._stop_event = threading.Event()

    def start_listening(self):
        """Start listening to the WebSocket."""
        self._stop_event.clear()
        self.transcription = []

        try:
            print("Connecting to WebSocket...")
            self.socket = connect(
                self.url,
                additional_headers={"Authorization": f"Token {self.token}"},
            )
            print("WebSocket connected.")

            # Start processing responses
            self._response_thread = threading.Thread(target=self.process_response, daemon=True)
            self._response_thread.start()

        except Exception as e:
            print(f"Error connecting to WebSocket: {e}")

    def stop_listening(self):
        """Stop listening and return transcription."""
        self._stop_event.set()
        if self.socket:
            self.socket.close()
        if hasattr(self, "_response_thread"):
            self._response_thread.join()
        return " ".join(self.transcription)

    def process_response(self):
        """Process incoming audio transcription."""
        print("Processing WebSocket responses...")
        try:
            while not self._stop_event.is_set():
                message = self.socket.recv()
                if message:
                    print(f"WebSocket response: {message}")
                    result = self._parse_transcription(message)
                    if result:
                        print(f"Recognized: {result}")
                        self.transcription.append(result)
                else:
                    print("No response received from WebSocket.")
        except Exception as e:
            print(f"Error processing WebSocket response: {e}")

    def _parse_transcription(self, message):
        """Parse transcription from WebSocket response."""
        try:
            response = json.loads(message)
            alternatives = response.get("channel", {}).get("alternatives", [])
            if alternatives:
                return alternatives[0].get("transcript", "")
        except Exception as e:
            print(f"Error parsing transcription: {e}")
        return None


class MicrophoneStreamer:
    def __init__(self, stt: SpeechToText):
        self.stt = stt
        self._audio = pyaudio.PyAudio()
        self._stop_event = threading.Event()
        self.stream = None

    def start_streaming(self):
        """Start streaming audio from the microphone."""
        try:
            print("Initializing microphone...")
            self.stream = self._audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
            self._stop_event.clear()
            self.stt.start_listening()
            print("Streaming audio...")
            self._streaming_thread = threading.Thread(target=self._stream_audio, daemon=True)
            self._streaming_thread.start()
        except Exception as e:
            print(f"Error initializing microphone: {e}")

    def stop_streaming(self):
        """Stop streaming audio."""
        self._stop_event.set()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self._audio.terminate()
        return self.stt.stop_listening()

    def _stream_audio(self):
        """Stream audio data to the WebSocket."""
        try:
            while not self._stop_event.is_set():
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                print(f"Captured {len(data)} bytes from mic")
                self.stt.socket.send(data)
        except Exception as e:
            print(f"Error streaming audio: {e}")


# Example Usage
if __name__ == "__main__":
    stt = SpeechToText()
    mic_streamer = MicrophoneStreamer(stt)

    try:
        print("Listening for speech...")
        mic_streamer.start_streaming()
        input("Press Enter to stop listening...\n")
    finally:
        result = mic_streamer.stop_streaming()
        print(f"Transcribed Text: {result}")
