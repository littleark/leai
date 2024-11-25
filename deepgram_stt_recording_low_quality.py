import asyncio
import json
import os
import wave
from datetime import datetime
from typing import Optional
from urllib.parse import urlencode

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
        self.audio_config = {
            "format": pyaudio.paFloat32,
            "channels": 1,
            "rate": 16000,
            "chunk": 2048,
        }

        self.frames = []
        self.transcription = []

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
            "punctuate": "true"
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

        print(f"Audio saved to {filename}")

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

async def main():
    transcriber = DeepgramTranscriber()

    # Record and transcribe for 5 seconds
    await transcriber.record_and_transcribe(duration=5)

    # Save the audio file
    transcriber.save_audio()

if __name__ == "__main__":
    asyncio.run(main())
