import os
import threading
import json
from dotenv import load_dotenv
from websockets.sync.client import connect
import pyaudio

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"Device {i}: {info['name']}, Sample rate: {info['defaultSampleRate']}, Channels: {info['maxInputChannels']}")

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

    def send_config_message(self):
        """Send configuration to the WebSocket to ensure the correct setup."""
        configure_message = {
            "type": "Configure",
            "features": {
                "speech_recognition": {
                    "language": "en-US",
                    "encoding": "linear16",  # Assuming 16-bit linear PCM encoding
                    "sample_rate": RATE,  # Adjust to your microphone sample rate
                    "channels": CHANNELS,  # Typically 1 for mono audio
                }
            }
        }
        self.socket.send(json.dumps(configure_message))
        print("Configure message sent.")


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

            self.send_config_message()

            # Send the Configure message with correct fields


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
                        print("No transcription found in the response.")
                else:
                    print("No response received from WebSocket.")
        except Exception as e:
            print(f"Error processing WebSocket response: {e}")


    def process_response3(self):
        """Process incoming audio transcription."""
        print("Processing WebSocket responses...")
        try:
            while not self._stop_event.is_set():
                message = self.socket.recv()
                if message:
                    print(f"WebSocket response: {message}")
                    try:
                        response = json.loads(message)
                        message_type = response.get("type")

                        # Handle different message types
                        if message_type == "Error":
                            print(f"Error from Deepgram: {response['description']}")
                        elif message_type == "Transcript":
                            result = self._parse_transcription(response)
                            if result:
                                print(f"Recognized: {result}")
                                self.transcription.append(result)
                        elif message_type in ["KeepAlive", "Finalize", "Sync"]:
                            # These types don't contain transcription data, so we can just skip them
                            pass
                        else:
                            print(f"Unexpected message type: {message_type}")

                    except json.JSONDecodeError:
                        print("Error decoding response.")
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
                transcript = alternatives[0].get("transcript", "")
                is_final = response.get("is_final", False)
                if is_final:
                    print(f"Final transcript: {transcript}")
                    return transcript
                else:
                    print(f"Partial transcript: {transcript}")
                    return transcript  # Keep the partial transcript, for debugging purposes
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
                try:
                    # Send audio data to the WebSocket
                    self.stt.socket.send(data)
                    print(f"Sent {len(data)} bytes to WebSocket.")
                except Exception as e:
                    print(f"Error sending data to WebSocket: {e}")
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
