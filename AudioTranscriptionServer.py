import asyncio
from models import RAGState
from fastapi import WebSocket, WebSocketDisconnect
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
    def __init__(self, deepgram_api_key=None, audio_save_dir='audio_chunks', client_id=None, connection_manager=None):
        self.client_id = client_id
        self.connection_manager = connection_manager

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

        self.current_websocket = websocket

        try:
            # Initialize Deepgram client
            print("Initializing Deepgram client")
            deepgram = DeepgramClient(self.deepgram_api_key)

            # Create Deepgram WebSocket connection
            self.dg_connection = deepgram.listen.websocket.v("1")

            async def wrapped_on_message(*args, **kwargs):
                try:
                    await self.on_message(*args, **kwargs)
                except Exception as e:
                    print(f"Error in wrapped_on_message: {e}")

            # Setup Deepgram event handlers
            self.dg_connection.on(LiveTranscriptionEvents.Open, self.on_open)
            self.dg_connection.on(LiveTranscriptionEvents.Transcript,
                lambda *args, **kwargs: self.loop.create_task(wrapped_on_message(*args, **kwargs)))
            self.dg_connection.on(LiveTranscriptionEvents.Metadata, self.on_metadata)
            self.dg_connection.on(LiveTranscriptionEvents.Error, self.on_error)
            self.dg_connection.on(LiveTranscriptionEvents.Close, self.on_close)
            self.dg_connection.on(LiveTranscriptionEvents.SpeechStarted, self.on_speech_started)
            self.dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, self.on_utterance_end)
            self.dg_connection.on(LiveTranscriptionEvents.Unhandled, self.on_unhandled)

            options: LiveOptions = LiveOptions(
                model="nova-2",
                language="en-US",
                smart_format=True,
                encoding="linear16",
                channels=self.audio_config['channels'],
                sample_rate=self.audio_config['rate'],
                interim_results=True,
                utterance_end_ms="1000",
                vad_events=True,
                endpointing=300,
            )

            addons = {
                "no_delay": "true"
            }

            # Start Deepgram connection
            if not self.dg_connection.start(options, addons=addons):
                self.logger.error("Failed to start Deepgram connection")
                return

            try:
                while True:
                    try:
                        data = await websocket.receive()
                        if data.get("type") == "websocket.receive":
                            if "bytes" in data:
                                audio_data = data["bytes"]
                                print(f"Received audio chunk of size: {len(audio_data)}")
                                self.dg_connection.send(audio_data)
                    except WebSocketDisconnect:
                        self.logger.info("Client disconnected normally")
                        break
                    except Exception as e:
                        self.logger.error(f"Error receiving message: {str(e)}")
                        break
            except Exception as e:
                self.logger.error(f"Error in message loop: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error in WebSocket handling, client {client_id}: {str(e)}")
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
        # Get the client's state if needed
        client_state = self.connection_manager.get_state(self.client_id) if self.client_id else None
        reader_name = client_state.reader_name if client_state else "Lucy"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.chat_api_url,
                json={"message": text, "reader_name": reader_name}
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
