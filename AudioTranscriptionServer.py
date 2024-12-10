import asyncio
import websockets
from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)
import os
import logging
import uuid
import wave
import pyaudio;
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

        self.chat_api_url = "http://localhost:8000/chat"

        self.is_finals = []

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.audio_save_dir = audio_save_dir
        os.makedirs(self.audio_save_dir, exist_ok=True)

    async def start_server(self, host='localhost', port=8765):
        """Start WebSocket server to receive audio and forward to Deepgram"""
        server = await websockets.serve(
            self.handle_client_websocket,
            host,
            port
        )
        self.logger.info(f"WebSocket server started on ws://{host}:{port}")
        await server.wait_closed()

    async def handle_client_websocket(self, websocket, path):
        """Handle incoming WebSocket connection from client"""

        self.current_websocket = websocket

        try:
            # Initialize Deepgram client
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
            # self.dg_connection.on(LiveTranscriptionEvents.Transcript, self.on_message)
            self.dg_connection.on(LiveTranscriptionEvents.Transcript, lambda *args, **kwargs: self.loop.create_task(wrapped_on_message(*args, **kwargs)))
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

            # Forward audio from client to Deepgram
            try:
                chunk_count = 0
                async for message in websocket:
                    # Generate a unique filename for each audio chunk
                    # chunk_count += 1
                    # filename = os.path.join(
                    #     self.audio_save_dir,
                    #     f'audio_chunk_{chunk_count}_{uuid.uuid4().hex}.wav'
                    # )

                    # # Save the audio chunk as a proper WAV file
                    # try:
                    #     # save_wav_chunk(message, filename)
                    #     with wave.open(filename, 'wb') as wav_file:
                    #         # Set wav file parameters
                    #         wav_file.setnchannels(self.audio_config["channels"])  # mono
                    #         wav_file.setsampwidth(pyaudio.get_sample_size(self.audio_config["format"]))  # 16-bit
                    #         wav_file.setframerate(self.audio_config["rate"])  # 16kHz
                    #         wav_file.writeframes(message)

                    #     # self.logger.info(f"Saved audio chunk to {filename}")
                    # except Exception as save_error:
                    #     self.logger.error(f"Error saving audio chunk: {save_error}")

                    # Send the message to Deepgram
                    # print(f"Sending message to Deepgram: {len(message)}")
                    self.dg_connection.send(message)

            except websockets.exceptions.ConnectionClosed:
                self.logger.info("Client WebSocket connection closed")

            finally:
                # Finish Deepgram connection
                self.current_websocket = None
                self.dg_connection.finish()

        except Exception as e:
            self.logger.error(f"Error in WebSocket handling: {e}")

    def on_open(self, open_event=None, connection=None, **kwargs):
        print(f"Connection Open: {open_event}")
        self.logger.info(f"Connection opened: {open_event}")

    def on_close(self, close_event=None, connection=None, **kwargs):
        print("Connection Closed")
        # self.concatenate_wav_files(
        #     audio_save_dir,
        #     os.path.join(audio_save_dir, 'full_recording.wav')
        # )

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
                    # Send the final utterance back to the browser
                    # chat_response = await self.send_to_chat(utterance)

                    # Send both transcription and chat response to client
                    if self.current_websocket:
                        await self.current_websocket.send(json.dumps({
                            "type": "final_transcript",
                            "transcript": utterance,
                            "is_final": True,
                            "speech_final": True
                        }))

                    self.is_finals = []
                    self.dg_connection.finish()
                else:
                    print(f"Is Final: {sentence}")
            else:
                print(f"Interim Results: {sentence}")
                if self.current_websocket:
                    await self.current_websocket.send(json.dumps({
                        "type": "interim_transcript",
                        "transcript": sentence
                    }))

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

    def concatenate_wav_files(self, input_dir, output_file):
        """
        Concatenate all WAV files in a directory into a single WAV file.

        Args:
            input_dir (str): Directory containing WAV files to concatenate
            output_file (str): Path to the output concatenated WAV file
        """
        # Get list of WAV files, sorted by creation time
        wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
        wav_files.sort(key=lambda x: os.path.getctime(os.path.join(input_dir, x)))

        if not wav_files:
            print("No WAV files found to concatenate.")
            return

        # Read the first file to get parameters
        with wave.open(os.path.join(input_dir, wav_files[0]), 'rb') as first_wav:
            params = first_wav.getparams()

        # Open output file
        with wave.open(output_file, 'wb') as outfile:
            outfile.setparams(params)

            # Append frames from each input file
            for wav_file in wav_files:
                filepath = os.path.join(input_dir, wav_file)
                with wave.open(filepath, 'rb') as w:
                    outfile.writeframes(w.readframes(w.getnframes()))

        print(f"Concatenated {len(wav_files)} WAV files into {output_file}")

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

        # Create an event loop and run the server
        server = await websockets.serve(
            transcription_server.handle_client_websocket,
            'localhost',
            8765
        )

        print(f"WebSocket server started on ws://localhost:8765")
        await server.wait_closed()

    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    asyncio.run(main())
