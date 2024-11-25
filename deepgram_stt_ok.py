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
import sys
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
            return await websockets.connect(url, extra_headers=extra_headers)
        except Exception as e:
            print(f"Failed to connect to Deepgram: {e}")
            return None

    async def record_and_transcribe(self):
        """Record audio and transcribe, stopping on silence or manual interrupt"""
        print("\nInitializing audio...")
        self.list_audio_devices()

        print("\nRecording... Press Enter to stop.")
        print("Recording will also stop after 2 seconds of silence.")

        # Start listener for manual stop
        listener_thread = threading.Thread(target=self.manual_stop_listener, daemon=True)
        listener_thread.start()

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

        # Connect to Deepgram
        websocket = await self.connect_websocket()
        if not websocket:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            return "Failed to connect to Deepgram"

        try:
            self.frames = []
            self.transcription = []

            while not self.stop_flag:
                # Read raw audio data
                data = stream.read(self.audio_config["chunk"], exception_on_overflow=False)

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
                await websocket.send(data)

                # Try to receive transcription
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    json_response = json.loads(response)
                    if "channel" in json_response:
                        transcript = json_response["channel"]["alternatives"][0]["transcript"]
                        if transcript.strip():
                            print(f"\nTranscribed: {transcript}")
                            self.transcription.append(transcript)
                except asyncio.TimeoutError:
                    continue

        except Exception as e:
            print(f"\nError during recording/transcription: {e}")

        finally:
            print("\nCleaning up...")
            stream.stop_stream()
            stream.close()
            audio.terminate()
            await websocket.close()

            final_transcription = " ".join(self.transcription)
            print(f"\nFinal transcription: {final_transcription}")
            return final_transcription

    def manual_stop_listener(self):
        """Wait for Enter key to manually stop recording"""
        input()
        print("\nManual stop triggered.")
        self.stop_flag = True

async def main():
    transcriber = DeepgramTranscriber()
    await transcriber.record_and_transcribe()
    transcriber.save_audio()

if __name__ == "__main__":
    asyncio.run(main())
