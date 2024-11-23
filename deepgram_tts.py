import json
import os
import threading
import time
from dotenv import load_dotenv
from websockets.sync.client import connect
import pyaudio
import queue

load_dotenv()

TIMEOUT = 0.050
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 8000

DEFAULT_URL = f"wss://api.deepgram.com/v1/speak?encoding=linear16&sample_rate={RATE}"
DEFAULT_TOKEN = os.environ.get("DEEPGRAM_API_KEY", None)


class Speaker:
    def __init__(self, rate=RATE, chunk=CHUNK, channels=CHANNELS):
        self._audio = pyaudio.PyAudio()
        self._chunk = chunk
        self._rate = rate
        self._format = FORMAT
        self._channels = channels
        self._exit = threading.Event()
        self._queue = queue.Queue()
        self._stream = None
        self._thread = None

    def start(self):
        self._stream = self._audio.open(
            format=self._format,
            channels=self._channels,
            rate=self._rate,
            input=False,
            output=True,
            frames_per_buffer=self._chunk,
        )
        self._exit.clear()
        self._thread = threading.Thread(
            target=self._playback, args=(self._queue, self._stream, self._exit), daemon=True
        )
        self._thread.start()

    def stop(self):
        self._exit.set()
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._thread:
            self._thread.join()
            self._thread = None

    def play(self, data):
        self._queue.put(data)

    def _playback(self, audio_queue, stream, exit_event):
        while not exit_event.is_set():
            try:
                data = audio_queue.get(timeout=TIMEOUT)
                stream.write(data)
            except queue.Empty:
                pass


class TextToSpeech:
    def __init__(self, url=DEFAULT_URL, token=DEFAULT_TOKEN):
        self.url = url
        self.token = token
        self.speaker = Speaker()
        self.socket = None
        self._receiver_thread = None
        self._exit_event = threading.Event()

    def play(self, text):
        """Plays the given text."""
        if not self.socket:
            self.socket = connect(
                self.url, additional_headers={"Authorization": f"Token {self.token}"}
            )
            self._start_receiver()

        # Send text chunk by chunk
        for chunk in self._chunk_text(text):
            self.socket.send(json.dumps({"type": "Speak", "text": chunk}))
            time.sleep(0.5)  # Allow time for playback to process

        # Ensure playback finishes
        self.socket.send(json.dumps({"type": "Flush"}))

    def stop(self):
        """Stops playback."""
        self._exit_event.set()
        if self.socket:
            self.socket.send(json.dumps({"type": "Close"}))
            self.socket.close()
            self.socket = None
        self.speaker.stop()

    def _start_receiver(self):
        """Starts the audio receiver thread."""
        self.speaker.start()
        self._receiver_thread = threading.Thread(target=self._receiver, daemon=True)
        self._receiver_thread.start()

    def _receiver(self):
        """Handles receiving audio data."""
        try:
            while not self._exit_event.is_set():
                message = self.socket.recv()
                if isinstance(message, bytes):
                    self.speaker.play(message)
        except Exception as e:
            print(f"Receiver error: {e}")
        finally:
            self.speaker.stop()

    def _chunk_text(self, text, max_length=100):
        """Splits text into smaller chunks for better API handling."""
        import re

        sentences = re.split(r"(?<=[.!?]) +", text)
        chunks = []

        for sentence in sentences:
            while len(sentence) > max_length:
                split_point = sentence[:max_length].rfind(" ")
                if split_point == -1:
                    split_point = max_length
                chunks.append(sentence[:split_point])
                sentence = sentence[split_point:].lstrip()
            chunks.append(sentence)

        return chunks


# Example Usage
if __name__ == "__main__":
    tts = TextToSpeech()
    try:
        tts.play(
            "Hello, this is a test of the Deepgram text-to-speech service. "
            "The text-to-speech should play everything, including punctuation and long sentences."
        )
        input("Press Enter to stop playback...")
    finally:
        tts.stop()
