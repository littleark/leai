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
import threading
import time

load_dotenv()

class DeepgramTranscriber:
    def __init__(self):
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY not found in environment variables")

        self.websocket_url = "wss://api.deepgram.com/v1/listen"

        # Simplified audio configuration
        self.audio_config = {
            "format": pyaudio.paInt16,  # Using 16-bit integers
            "channels": 1,
            "rate": 16000,             # Standard rate for speech
            "chunk": 1024,             # Smaller chunks for better testing
            "input_device_index": None
        }

        self.frames = []
        self.transcription = []
        self.stop_flag = False
        self.last_audio_time = time.time()
        self.websocket = None
        self.stream = None
        self.audio = None

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
        """Start listening and transcribing"""
        print("\nInitializing audio...")
        self.list_audio_devices()

        print("\nListening...")

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

        # Connect to Deepgram
        await self.connect_websocket()
        if not self.websocket:
            self.stop_flag = True
            return "Failed to connect to Deepgram"

        self.frames = []
        self.transcription = []
        self.stop_flag = False

        while not self.stop_flag:
            # Read raw audio data
            data = self.stream.read(self.audio_config["chunk"], exception_on_overflow=False)

            # Append raw audio frames for WAV saving
            self.frames.append(data)

            # Analyze audio level
            audio_array = np.frombuffer(data, dtype=np.int16)
            peak = np.max(np.abs(audio_array))
            print(f"\rAudio Level: {peak}", end='')

            # Reset the silence timer if sound is detected
            if peak > 500:  # Threshold for sound detection
                self.last_audio_time = time.time()

            # Stop if silence lasts for 2 seconds
            if time.time() - self.last_audio_time > 2:
                print("\nSilence detected. Stopping recording...")
                self.stop_flag = True
                break

            # Send raw data to Deepgram for transcription
            await self.websocket.send(data)

            # Try to receive transcription
            try:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=0.1)
                json_response = json.loads(response)
                if "channel" in json_response:
                    transcript = json_response["channel"]["alternatives"][0]["transcript"]
                    if transcript.strip():
                        print(f"\nTranscribed: {transcript}")
                        self.transcription.append(transcript)
            except asyncio.TimeoutError:
                continue

        return " ".join(self.transcription)

    def stop_listening(self):
        """Stop listening manually"""
        print("\nManual stop triggered.")
        self.stop_flag = True
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        if self.websocket:
            asyncio.run(self.websocket.close())
        return " ".join(self.transcription)

def listen_and_transcribe_sync():
    """Run the async function in a blocking way for Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(transcriber.listen_and_transcribe())

async def main():
    transcriber = DeepgramTranscriber()
    await transcriber.listen_and_transcribe()
    transcriber.save_audio()

if __name__ == "__main__":
    transcriber = DeepgramTranscriber()
    text = listen_and_transcribe_sync()
    print(text)
    # asyncio.run(main())
