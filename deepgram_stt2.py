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

        # Updated audio configuration for better quality
        self.audio_config = {
            "format": pyaudio.paFloat32,
            "channels": 1,
            "rate": 44100,  # Increased sample rate
            "chunk": 4096,  # Larger chunk size
            "input_device_index": None  # Will be set during device selection
        }

        self.frames = []
        self.transcription = []

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

    def process_audio_chunk(self, audio_data):
        """Process audio chunk to improve quality"""
        # Convert byte data to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.float32)

        # Apply noise gate
        noise_gate = 0.02  # Adjust this threshold as needed
        audio_array[abs(audio_array) < noise_gate] = 0

        # Normalize audio (make quiet sounds louder)
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))

        # Apply soft knee compression
        threshold = 0.5
        ratio = 0.3
        makeup_gain = 1.5

        mask = np.abs(audio_array) > threshold
        audio_array[mask] = np.sign(audio_array[mask]) * (
            threshold + (np.abs(audio_array[mask]) - threshold) * ratio
        )
        audio_array *= makeup_gain

        # Ensure we don't exceed [-1, 1] range
        audio_array = np.clip(audio_array, -1, 1)

        # Print audio levels for monitoring
        max_level = np.max(np.abs(audio_array))
        self.print_audio_level(max_level)

        return audio_array.tobytes()

    def print_audio_level(self, level):
        """Print a visual representation of audio level"""
        bars = int(level * 50)
        print(f"\rAudio Level: {'|' * bars}{' ' * (50 - bars)} {level:.2f}", end='')

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
        """Record audio and get transcription for specified duration"""
        print(f"Starting recording for {duration} seconds...")

        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.audio_config["format"],
            channels=self.audio_config["channels"],
            rate=self.audio_config["rate"],
            input=True,
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
            # Clear previous data
            self.frames = []
            self.transcription = []

            # Set end time
            end_time = asyncio.get_event_loop().time() + duration

            while asyncio.get_event_loop().time() < end_time:
                # Read audio data
                data = stream.read(self.audio_config["chunk"], exception_on_overflow=False)
                self.frames.append(data)

                # Send to Deepgram
                try:
                    await websocket.send(data)

                    # Try to receive transcription with timeout
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        json_response = json.loads(response)
                        if "channel" in json_response:
                            transcript = json_response["channel"]["alternatives"][0]["transcript"]
                            if transcript.strip():
                                print(f"Transcribed: {transcript}")
                                self.transcription.append(transcript)
                    except asyncio.TimeoutError:
                        continue

                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed")
                    break

                await asyncio.sleep(0)  # Allow other tasks to run

        except Exception as e:
            print(f"Error during recording/transcription: {e}")

        finally:
            print("Cleaning up...")
            # Cleanup
            stream.stop_stream()
            stream.close()
            audio.terminate()
            await websocket.close()

            final_transcription = " ".join(self.transcription)
            print(f"\nFinal transcription: {final_transcription}")
            return final_transcription

    async def record_and_transcribe2(self, duration: int = 5):
        """Record audio and get transcription for specified duration"""
        print("\nInitializing audio...")
        self.list_audio_devices()

        print(f"\nStarting recording for {duration} seconds...")
        print("Speak clearly and not too far from the microphone")
        print("\nAudio level monitor (should show movement when speaking):")

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
            end_time = asyncio.get_event_loop().time() + duration

            while asyncio.get_event_loop().time() < end_time:
                # Read and process audio
                data = stream.read(self.audio_config["chunk"], exception_on_overflow=False)
                processed_data = self.process_audio_chunk(data)
                self.frames.append(processed_data)

                # Send to Deepgram
                try:
                    await websocket.send(processed_data)

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

                except websockets.exceptions.ConnectionClosed:
                    print("\nWebSocket connection closed")
                    break

                await asyncio.sleep(0)

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

async def main():
    transcriber = DeepgramTranscriber()

    # Record and transcribe for 5 seconds
    await transcriber.record_and_transcribe(duration=10)  # Increased to 10 seconds

    # Save the audio file
    transcriber.save_audio()

if __name__ == "__main__":
    asyncio.run(main())
