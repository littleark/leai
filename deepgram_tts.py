import json
import os
import threading
import queue
import websockets
from websockets.sync.client import connect
import pyaudio
from dotenv import load_dotenv
import wave

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
        self._audio.terminate()

    def play(self, data):
        self._queue.put(data)

    def _playback(self, audio_queue, stream, exit_event):
        while not exit_event.is_set():
            try:
                data = audio_queue.get(timeout=TIMEOUT)
                stream.write(data)
            except queue.Empty:
                pass

def text_to_speech(story):
    print(f"Connecting to {DEFAULT_URL}")
    socket = connect(
        DEFAULT_URL,
        additional_headers={"Authorization": f"Token {DEFAULT_TOKEN}"}
    )
    exit_event = threading.Event()
    speaker = Speaker()
    speaker.start()  # Uncomment this line to start audio playback

    def receiver():
        try:
            while not exit_event.is_set():
                try:
                    message = socket.recv()

                    if message is None:
                        continue

                    if isinstance(message, str):
                        print(f"Received status: {message}")
                    elif isinstance(message, bytes):
                        speaker.play(message)

                except Exception as e:
                    print(f"Receive error: {e}")
                    break
        except Exception as e:
            print(f"Receiver error: {e}")
        finally:
            speaker.stop()

    # Start receiver in a separate thread
    receiver_thread = threading.Thread(target=receiver, daemon=True)
    receiver_thread.start()

    try:
        # Send text inputs
        for text_input in story:
            print(f"Sending: {text_input}")
            socket.send(json.dumps({"type": "Speak", "text": text_input}))

        # Flush and close
        print("Flushing...")
        socket.send(json.dumps({"type": "Flush"}))

        # Wait for a bit to ensure all audio is played
        import time
        time.sleep(5)  # Give time for audio to play

    except Exception as e:
        print(f"Error during text-to-speech: {e}")

    finally:
        # Cleanup
        exit_event.set()
        socket.close()
        receiver_thread.join()
        speaker.stop()

def text_to_speech_buffer2(text_input):
    """Version of text_to_speech that returns audio data instead of playing it"""
    print(f"Connecting to {DEFAULT_URL}")
    audio_buffer = bytearray()

    try:
        socket = connect(
            DEFAULT_URL,
            additional_headers={"Authorization": f"Token {DEFAULT_TOKEN}"}
        )

        # Send the text input
        if isinstance(text_input, (list, tuple)):
            text_inputs = text_input
        else:
            text_inputs = [text_input]

        # Send each text input separately
        for text in text_inputs:
            print(f"Sending: {text}")
            socket.send(json.dumps({"type": "Speak", "text": text}))

        # Flush
        print("Flushing...")
        socket.send(json.dumps({"type": "Flush"}))

        # Receive audio data
        while True:
            try:
                message = socket.recv()
                print('message', len(message))
                if message is None:
                    print('message is None')
                    break

                if isinstance(message, str):
                    print(f"Received status: {message}")
                    # Check if this is the end of the stream
                    if '"type":"Flushed"' in message:
                        print('Flushed')
                        break
                elif isinstance(message, bytes):
                    audio_buffer.extend(message)

            except Exception as e:
                print(f"Receive error: {e}")
                break

    except Exception as e:
        print(f"Error during text-to-speech: {e}")
        return None

    finally:
        print('closing socket')
        socket.close()

    print('audio_buffer', len(audio_buffer))
    return bytes(audio_buffer)

def text_to_speech_buffer(text_input):
    """Version of text_to_speech that returns audio data instead of playing it"""
    print(f"Connecting to {DEFAULT_URL}")
    audio_buffer = bytearray()
    socket = None

    try:
        socket = connect(
            DEFAULT_URL,
            additional_headers={"Authorization": f"Token {DEFAULT_TOKEN}"},
            close_timeout=1
        )

        # Handle both single strings and lists of strings
        if isinstance(text_input, (list, tuple)):
            text_inputs = text_input
        else:
            text_inputs = [text_input]

        # Send all text inputs first
        for text in text_inputs:
            print(f"Sending: {text}")
            socket.send(json.dumps({"type": "Speak", "text": text}))

        # Send flush command
        print("Flushing...")
        socket.send(json.dumps({"type": "Flush"}))

        # Collect all messages until we get the "Flushed" message
        got_metadata = False
        while True:
            message = socket.recv()

            if message is None:
                continue

            if isinstance(message, str):
                print(f"Received status: {message}")
                if '"type":"Metadata"' in message:
                    got_metadata = True
                if '"type":"Flushed"' in message:
                    print("Received flush signal, stream complete")
                    # Close the connection immediately after receiving Flushed message
                    if socket:
                        print('closing socket')
                        try:
                            socket.close()
                            socket = None
                            print("Closed socket")
                        except Exception as e:
                            print(f"Error closing socket: {e}")
                    break
            elif isinstance(message, bytes) and got_metadata:
                print(f"Received audio chunk: {len(message)} bytes")
                audio_buffer.extend(message)

    except Exception as e:
        print(f"Error during text-to-speech: {e}")
        return None

    finally:
        if socket:
            if socket:
                print("if socket -> socket.close()")
                try:
                    socket.close()
                except Exception as e:
                    print(f"Error closing socket in finally block: {e}")

    final_audio = bytes(audio_buffer)
    print(f"Total audio length: {len(final_audio)} bytes")
    return final_audio

def main():
    # Test with a single string
    single_text = "The sun"
    audio_data = text_to_speech_buffer(single_text)

    if audio_data and len(audio_data) > 0:
        # Save to a WAV file for testing
        with wave.open("test_output.wav", 'wb') as wave_file:
            wave_file.setnchannels(1)  # mono
            wave_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wave_file.setframerate(48000)  # sample rate
            wave_file.writeframes(audio_data)
        print(f"Saved audio file with {len(audio_data)} bytes")
    else:
        print("No audio data received")

if __name__ == "__main__":
    main()
