import asyncio
from fastapi import WebSocket
from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)
import os
import logging
import uuid
import wave
import pyaudio
import aiohttp
import json

from dotenv import load_dotenv
load_dotenv()

class AudioTranscriptionServer:
    def __init__(self, deepgram_api_key=None, audio_save_dir='audio_chunks'):
        # Use API key from environment or parameter
        self.deepgram_api_key = deepgram_api_key or os.getenv('DEEPGRAM_API_KEY')

        self.audio_config = {
            "format": pyaudio.paInt16,
            "channels": 2,
            "rate": 44100,
        }

        self.loop = asyncio.get_event_loop()

        self.chat_api_url = "http://localhost:7860/chat"

        self.is_finals = []

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.audio_save_dir = audio_save_dir
        os.makedirs(self.audio_save_dir, exist_ok=True)

        self.current_websocket = None
        self.dg_connection = None

    async def handle_fastapi_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection using FastAPI's WebSocket"""
        await websocket.accept()
        self.current_websocket = websocket

        try:
            # Initialize Deepgram client
            print("Initializing Deepgram client")
            deepgram = DeepgramClient(self.deepgram_api_key)

            # Create Deepgram WebSocket connection
            self.dg_connection = deepgram.listen.websocket.v("1")

            # ... (rest of the setup code remains the same)

            try:
                while True:
                    try:
                        message = await websocket.receive_bytes()
                        self.dg_connection.send(message)
                    except WebSocketDisconnect:
                        self.logger.info("Client disconnected normally")
                        break
                    except ConnectionClosed:
                        self.logger.info("WebSocket connection closed")
                        break
                    except Exception as e:
                        self.logger.error(f"Error receiving message: {str(e)}")
                        break

            except Exception as e:
                self.logger.error(f"Error in WebSocket message loop: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error in WebSocket handling: {str(e)}")
            import traceback
            traceback.print_exc()

        finally:
            try:
                if self.dg_connection:
                    self.dg_connection.finish()
                self.current_websocket = None
                self.logger.info("Cleaned up WebSocket connection")
            except Exception as e:
                self.logger.error(f"Error during cleanup: {str(e)}")

    def on_open(self, open_event=None, connection=None, **kwargs):
        print(f"Connection Open: {open_event}")
        self.logger.info(f"Connection opened: {open_event}")

    def on_close(self, close_event=None, connection=None, **kwargs):
        print("Connection Closed")

    def on_metadata(self, client=None, metadata=None, **kwargs):
        print(f"Metadata: {metadata}")

    def on_speech_started(self, client=None, speech_started=None, **kwargs):
        print("Speech Started")

    def on_utterance_end(self, client=None, utterance_end = None, **kwargs):
        print("Utterance End")
        if len(self.is_finals) > 0:
            utterance = " ".join(self.is_finals)
            print(f"Utterance End: {utterance}")
            self.is_finals = []

    def on_error(self, error, connection=None,**kwargs):
        print(f"Handled Error: {error}")

    def on_unhandled(self, unhandled, connection=None,**kwargs):
        print(f"Unhandled Websocket Message: {unhandled}")

    async def on_message(self, client=None, result=None, **kwargs):
        try:
            if result is None:
                print("No result received")
                return

            # Safely access transcript
            try:
                sentence = result.channel.alternatives[0].transcript
            except (AttributeError, IndexError) as e:
                print(f"Error accessing transcript: {e}")
                return

            print("ON MESSAGE")

            if len(sentence) == 0:
                print('Sentence length is 0')
                return

            if result.is_final:
                print(f"Message: {result.to_json()}")
                self.is_finals.append(sentence)

                if result.speech_final:
                    utterance = " ".join(self.is_finals)
                    print(f"Speech Final: {utterance}")

                    if self.current_websocket:
                        await self.current_websocket.send_json({
                            "type": "final_transcript",
                            "transcript": utterance,
                            "is_final": True,
                            "speech_final": True
                        })

                    self.is_finals = []
                    self.dg_connection.finish()
                else:
                    print(f"Is Final: {sentence}")
            else:
                print(f"Interim Results: {sentence}")
                if self.current_websocket:
                    await self.current_websocket.send_json({
                        "type": "interim_transcript",
                        "transcript": sentence
                    })

        except Exception as e:
            print(f"Error in on_message: {e}")
            import traceback
            traceback.print_exc()

    async def send_to_chat(self, text):
        """Send transcribed text to chat endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.chat_api_url,
                json={"message": text, "reader_name": "Lucy"}  # Adjust reader_name as needed
            ) as response:
                return await response.json()

    def save_wav_chunk(self, message, filename, channels=1, sample_width=2, sample_rate=44100):
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(message)

async def main():
    try:
        # Create and start the audio transcription server
        transcription_server = AudioTranscriptionServer()

        await transcription_server.start_server()

        # Create an event loop and run the server
        # server = await websockets.serve(
        #     transcription_server.handle_client_websocket,
        #     'localhost',
        #     8765
        # )

        # print(f"WebSocket server started on ws://localhost:8765")
        # await server.wait_closed()

    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    asyncio.run(main())
