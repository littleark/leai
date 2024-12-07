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

DEFAULT_URL = f"wss://api.deepgram.com/v1/speak?encoding=linear16&model=aura-luna-en&sample_rate={RATE}"
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
        self._audio_buffer = bytearray()

    async def generate(self, text):
        """Generates audio data for the given text and returns it."""
        try:
            self._audio_buffer = bytearray()

            # Create new socket connection for each generation
            socket = connect(
                self.url,
                additional_headers={"Authorization": f"Token {self.token}"},
                open_timeout=10,  # 10-second timeout for connection
                close_timeout=10  # 10-second timeout for closing
            )

            try:
                # Send the text
                socket.send(json.dumps({"type": "Speak", "text": text}))

                finished = False
                timeout_duration = 30  # Max 30 seconds to receive the audio
                start_time = time.time()

                while not finished:
                    # Check overall timeout
                    if time.time() - start_time > timeout_duration:
                        print("Error: Audio generation timed out.")
                        break

                    try:
                        message = socket.recv()

                        if isinstance(message, bytes):
                            self._audio_buffer.extend(message)
                        elif isinstance(message, str):
                            try:
                                status = json.loads(message)
                                if status.get("type") == "Finished":
                                    finished = True
                                elif status.get("type") == "Error":
                                    print(f"Deepgram error: {status}")
                                    break
                            except json.JSONDecodeError:
                                print(f"Invalid JSON message: {message}")

                    except TimeoutError:
                        print("Socket timeout - ending stream")
                        break
                    except Exception as e:
                        print(f"Receive error: {e}")
                        break

                # If not finished, attempt a final flush
                if not finished:
                    try:
                        socket.send(json.dumps({"type": "Flush"}))
                        flush_message = socket.recv()
                        if isinstance(flush_message, bytes):
                            self._audio_buffer.extend(flush_message)
                    except Exception as e:
                        print(f"Flush error: {e}")

            finally:
                # Clean up socket
                try:
                    socket.send(json.dumps({"type": "Close"}))
                    socket.close()
                except:
                    pass

            # Convert buffer to bytes
            audio_data = bytes(self._audio_buffer)

            if len(audio_data) == 0:
                print("Warning: No audio data generated")
                return None

            print(f"Generated audio size: {len(audio_data)} bytes")
            return audio_data

        except Exception as e:
            print(f"Error generating audio: {e}")
            return None



    def play(self, text):
        """Plays the given text."""
        if not self.socket:
            self.socket = connect(
                self.url,
                additional_headers={"Authorization": f"Token {self.token}"}
            )
            self._start_receiver()

        # Send entire text at once
        self.socket.send(json.dumps({"type": "Speak", "text": text}))

        # Ensure playback finishes
        self.socket.send(json.dumps({"type": "Flush"}))

    def stop(self):
        """Stops playback."""
        self._exit_event.set()
        if self.socket:
            try:
                self.socket.send(json.dumps({"type": "Close"}))
                self.socket.close()
            except:
                pass
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
