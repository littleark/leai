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
        while True:
            try:
                device_index = int(input("Select input device by number (or press Enter for default): ").strip())
                if 0 <= device_index < num_devices:
                    self.audio_config["input_device_index"] = device_index
                    break
            except ValueError:
                self.audio_config["input_device_index"] = None
                break
            print("Invalid device number, try again.")

    def save_raw_audio(self, audio_data):
        """Save raw audio for debugging purposes"""
        with open("raw_audio.raw", "wb") as f:
            f.write(audio_data)
        print("\nRaw audio saved for analysis.")

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

    async def record_and_transcribe(self, duration: int = 5):
        """Record audio for debugging"""
        print("\nInitializing audio...")
        self.list_audio_devices()

        print(f"\nStarting recording for {duration} seconds...")
        print("Speak clearly and not too far from the microphone")
        print("\nAudio level monitor:")

        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.audio_config["format"],
            channels=self.audio_config["channels"],
            rate=self.audio_config["rate"],
            input=True,
            input_device_index=self.audio_config["input_device_index"],
            frames_per_buffer=self.audio_config["chunk"],
        )

        try:
            self.frames = []
            end_time = asyncio.get_event_loop().time() + duration

            while asyncio.get_event_loop().time() < end_time:
                # Read raw audio data
                data = stream.read(self.audio_config["chunk"], exception_on_overflow=False)

                # Save raw audio for analysis
                self.save_raw_audio(data)

                # Append raw audio frames for WAV saving
                self.frames.append(data)

                # Analyze audio level
                audio_array = np.frombuffer(data, dtype=np.int16)
                peak = np.max(np.abs(audio_array))
                print(f"\rAudio Level: {peak}", end='')

        except Exception as e:
            print(f"\nError during recording: {e}")

        finally:
            print("\nCleaning up...")
            stream.stop_stream()
            stream.close()
            audio.terminate()

    async def connect_websocket(self):
        """Dummy placeholder for transcription functionality"""
        print("\nWebSocket connection is disabled for this test. Focus on audio quality.")

async def main():
    transcriber = DeepgramTranscriber()

    # Record for 10 seconds for debugging
    await transcriber.record_and_transcribe(duration=10)

    # Save the audio file
    transcriber.save_audio()

if __name__ == "__main__":
    asyncio.run(main())
