import asyncio
import json
import os
import wave
from datetime import datetime
from typing import Optional
from urllib.parse import urlencode
import numpy as np
import pyaudio
import websockets
from dotenv import load_dotenv
import time

load_dotenv()

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", None)

class DeepgramTranscriber:
    def __init__(self):
        self.api_key = DEEPGRAM_API_KEY
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY not found in environment variables")

        self.websocket_url = "wss://api.deepgram.com/v1/listen"

        # Simplified audio configuration
        self.audio_config = {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": 16000,
            "chunk": 4096,
            "input_device_index": None,
        }

        self.frames = []
        self.transcription = []
        self.stop_flag = False
        self.websocket = None
        self.stream = None
        self.audio = None
        self.last_audio_time = time.time()

    async def connect_websocket(self):
        """Establish WebSocket connection with Deepgram."""
        extra_headers = {
            "Authorization": f"Token {self.api_key}",
        }
        print(f"extra_headers: {extra_headers}")

        params = {
            "encoding": "linear16",
            "sample_rate": self.audio_config["rate"],
            "channels": self.audio_config["channels"],
            "model": "general",
            "language": "en",
            "punctuate": "true",
            "endpointing": "500",  # Add endpointing for better sentence detection
        }

        url = f"{self.websocket_url}?{urlencode(params)}"
        print(f"Connecting to URL: {url}")

        try:
            self.websocket = await websockets.connect(url, additional_headers=extra_headers)
        except Exception as e:
            print(f"Failed to connect to Deepgram: {e}")
            self.websocket = None

    def save_audio(self, filename: Optional[str] = None):
        """Save recorded audio to WAV file"""
        if not self.frames:
            print("No audio data to save")
            return

        if filename is None:
            filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.audio_config["channels"])
            wf.setsampwidth(pyaudio.get_sample_size(self.audio_config["format"]))
            wf.setframerate(self.audio_config["rate"])
            wf.writeframes(b''.join(self.frames))

        print(f"\nAudio saved to {filename}")

    async def listen_and_transcribe(self):
        """Listen and transcribe in real-time."""
        self.stop_flag = False
        self.frames = []
        self.transcription = []

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.audio_config["format"],
            channels=self.audio_config["channels"],
            rate=self.audio_config["rate"],
            input=True,
            input_device_index=self.audio_config["input_device_index"],
            frames_per_buffer=self.audio_config["chunk"],
        )

        # Connect to Deepgram WebSocket
        await self.connect_websocket()
        if not self.websocket:
            return "Failed to connect to Deepgram"

        try:
            while not self.stop_flag:
                data = self.stream.read(self.audio_config["chunk"], exception_on_overflow=False)
                self.frames.append(data)

                # Detect silence
                audio_array = np.frombuffer(data, dtype=np.int16)
                peak = np.max(np.abs(audio_array))
                if peak > 500:
                    self.last_audio_time = time.time()

                if time.time() - self.last_audio_time > 2:
                    self.stop_flag = True
                    break

                await self.websocket.send(data)
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=0.1)
                    json_response = json.loads(response)
                    if "channel" in json_response:
                        transcript = json_response["channel"]["alternatives"][0]["transcript"]
                        if transcript.strip():
                            self.transcription.append(transcript)
                except asyncio.TimeoutError:
                    continue
        finally:
            await self.cleanup()
            return " ".join(self.transcription)

    async def stop_listening(self):
        """Manually stop listening."""
        self.stop_flag = True
        await self.cleanup()

    async def cleanup(self):
        """Clean up resources asynchronously."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        if self.websocket:
            await self.websocket.close()




class DeepgramTranscriberOld:
    def __init__(self):
            self.api_key = os.getenv("DEEPGRAM_API_KEY")
            if not self.api_key:
                raise ValueError("DEEPGRAM_API_KEY not found in environment variables")

            self.websocket_url = "wss://api.deepgram.com/v1/listen"

            # Simplified audio configuration
            self.audio_config = {
                "format": pyaudio.paInt16,
                "channels": 1,
                "rate": 16000,
                "chunk": 1024,
                "input_device_index": None,
            }

            self.frames = []
            self.transcription = []
            self.stop_flag = False
            self.websocket = None
            self.stream = None
            self.audio = None
            self.last_audio_time = time.time()

    def list_audio_devices(self):
        """List all available audio input devices"""
        audio = pyaudio.PyAudio()
        info = audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')

        print("\nAvailable Audio Input Devices:")
        print("-" * 30)

        for i in range(num_devices):
            device_info = audio.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:  # if it has input channels
                print(f"Device {i}: {device_info.get('name')}")
                print(f"  Max Input Channels: {device_info.get('maxInputChannels')}")
                print(f"  Default Sample Rate: {device_info.get('defaultSampleRate')}")
                print()

        audio.terminate()

        # Let user select device
        # while True:
        try:
            device_index = 0 # int(input("Select input device by number (or press Enter for default): ").strip())
            if 0 <= device_index < num_devices:
                self.audio_config["input_device_index"] = device_index
                # break
        except ValueError:
            self.audio_config["input_device_index"] = None
            # break
        print("Invalid device number, try again.")

    def save_audio(self, filename: Optional[str] = None):
        """Save recorded audio to WAV file"""
        if not self.frames:
            print("No audio data to save")
            return

        if filename is None:
            filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.audio_config["channels"])
            wf.setsampwidth(pyaudio.get_sample_size(self.audio_config["format"]))
            wf.setframerate(self.audio_config["rate"])
            wf.writeframes(b''.join(self.frames))

        print(f"\nAudio saved to {filename}")

    async def connect_websocket(self):
        """Establish WebSocket connection with Deepgram"""
        extra_headers = {
            "Authorization": f"Token {self.api_key}",
        }

        params = {
            "encoding": "linear16",
            "sample_rate": self.audio_config["rate"],
            "channels": self.audio_config["channels"],
            "model": "general",
            "language": "en",
            "punctuate": "true",
            "endpointing": "500",  # Add endpointing for better sentence detection
        }

        url = f"{self.websocket_url}?{urlencode(params)}"
        print(f"Connecting to URL: {url}")

        try:
            self.websocket = await websockets.connect(url, extra_headers=extra_headers)
        except Exception as e:
            print(f"Failed to connect to Deepgram: {e}")
            self.websocket = None

    async def listen_and_transcribe(self):
        """Listen and transcribe in real-time."""
        print("listening")
        self.stop_flag = False
        self.frames = []
        self.transcription = []

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.audio_config["format"],
            channels=self.audio_config["channels"],
            rate=self.audio_config["rate"],
            input=True,
            input_device_index=self.audio_config["input_device_index"],
            frames_per_buffer=self.audio_config["chunk"],
        )

        # Connect to Deepgram WebSocket
        await self.connect_websocket()
        if not self.websocket:
            return "Failed to connect to Deepgram"

        try:
            while not self.stop_flag:
                data = self.stream.read(self.audio_config["chunk"], exception_on_overflow=False)
                self.frames.append(data)

                # Detect silence
                audio_array = np.frombuffer(data, dtype=np.int16)
                peak = np.max(np.abs(audio_array))
                if peak > 500:
                    self.last_audio_time = time.time()

                if time.time() - self.last_audio_time > 2:
                    self.stop_flag = True
                    break

                await self.websocket.send(data)
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=0.1)
                    json_response = json.loads(response)
                    if "channel" in json_response:
                        transcript = json_response["channel"]["alternatives"][0]["transcript"]
                        if transcript.strip():
                            self.transcription.append(transcript)
                except asyncio.TimeoutError:
                    continue
        finally:
            self.cleanup()
            return " ".join(self.transcription)

    def stop_listening(self):
        """Manually stop listening."""
        self.stop_flag = True
        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        if self.websocket:
            asyncio.run(self.websocket.close())

def listen_and_transcribe_sync():
    """Run the async function in a blocking way for Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    text = loop.run_until_complete(transcriber.listen_and_transcribe())
    transcriber.save_audio()
    print(text)
    return text

async def main():
    transcriber = DeepgramTranscriber()
    await transcriber.listen_and_transcribe()
    transcriber.save_audio()

if __name__ == "__main__":
    transcriber = DeepgramTranscriber()
    text = listen_and_transcribe_sync()
    print(text)
    # asyncio.run(main())
