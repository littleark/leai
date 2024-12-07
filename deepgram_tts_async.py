import asyncio
import json
import os
import websockets
import pyaudio
from dotenv import load_dotenv

load_dotenv()

class AsyncTextToSpeech:
    def __init__(self,
                 url="wss://api.deepgram.com/v1/speak",
                 model="aura-luna-en",
                 token=None,
                 sample_rate=48000):
        """
        Initialize Deepgram Text-to-Speech client.

        :param url: Deepgram WebSocket URL
        :param model: TTS model to use
        :param token: Deepgram API token
        :param sample_rate: Audio sample rate
        """
        self.token = token or os.environ.get("DEEPGRAM_API_KEY")
        if not self.token:
            raise ValueError("Deepgram API token is required")

        self.url = f"{url}?encoding=linear16&model={model}&sample_rate={sample_rate}"
        self._pyaudio = pyaudio.PyAudio()
        self._stream = None

    async def generate(self, text, max_timeout=30):
        """
        Generate audio from text using Deepgram's WebSocket API.

        :param text: Text to convert to speech
        :param max_timeout: Maximum time to wait for audio generation
        :return: Bytes containing audio data
        """
        try:
            async with websockets.connect(
                self.url,
                additional_headers={"Authorization": f"Token {self.token}"}
            ) as websocket:
                # Send speak request
                await websocket.send(json.dumps({
                    "type": "Speak",
                    "text": text
                }))

                # Collect audio chunks
                audio_buffer = bytearray()
                start_time = asyncio.get_event_loop().time()

                # Track chunks to help detect end of stream
                last_chunk_size = 0
                consecutive_small_chunks = 0
                total_chunks = 0

                while True:
                    # Check for timeout
                    if asyncio.get_event_loop().time() - start_time > max_timeout:
                        print("Audio generation timed out")
                        break

                    try:
                        message = await websocket.recv()

                        # Only handle binary messages
                        if isinstance(message, bytes):
                            total_chunks += 1
                            audio_buffer.extend(message)

                            # Track chunk sizes to detect end of stream
                            if len(message) < 100:  # Small chunk threshold
                                consecutive_small_chunks += 1
                            else:
                                consecutive_small_chunks = 0

                            print(f"Received audio chunk {total_chunks}: {len(message)} bytes")

                            # Detect potential end of stream
                            if consecutive_small_chunks >= 3 or \
                               (total_chunks > 5 and len(message) == 0):
                                print("Detected potential end of audio stream")
                                break

                    except websockets.exceptions.ConnectionClosed:
                        print("WebSocket connection closed")
                        break
                    except Exception as e:
                        print(f"Error receiving message: {e}")
                        break

                # Explicitly send close message
                try:
                    await websocket.send(json.dumps({"type": "Close"}))
                except:
                    pass

                # Return audio data
                if audio_buffer:
                    print(f"Total audio generated: {len(audio_buffer)} bytes")
                    return bytes(audio_buffer)
                else:
                    print("No audio data generated")
                    return None

        except Exception as e:
            print(f"Error generating audio: {e}")
            return None

    def play(self, audio_data):
        """
        Play generated audio data.

        :param audio_data: Bytes containing audio to play
        """
        if not audio_data:
            print("No audio data to play")
            return

        # Open PyAudio stream
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=48000,
            output=True
        )

        # Play audio
        self._stream.write(audio_data)

        # Close stream
        self._stream.stop_stream()
        self._stream.close()

    async def text_to_speech(self, text):
        """
        Convenience method to generate and play audio in one call.

        :param text: Text to convert to speech
        """
        audio_data = await self.generate(text)
        if audio_data:
            self.play(audio_data)

# Example usage
async def main():
    tts = AsyncTextToSpeech()
    await tts.text_to_speech("Hello, this is a test of the Deepgram text-to-speech service.")

if __name__ == "__main__":
    asyncio.run(main())
