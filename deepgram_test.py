import json
import os
import threading
import asyncio
import queue
from dotenv import load_dotenv

import websockets
from websockets.sync.client import connect

import pyaudio

load_dotenv()

TIMEOUT = 0.050
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 8000

DEFAULT_URL = f"wss://api.deepgram.com/v1/speak?encoding=linear16&sample_rate={RATE}"
DEFAULT_TOKEN = os.environ.get("DEEPGRAM_API_KEY", None)

def main():
    print(f"Connecting to {DEFAULT_URL}")

    _socket = connect(
        DEFAULT_URL, additional_headers={"Authorization": f"Token {DEFAULT_TOKEN}"}
    )
    _exit = threading.Event()

    _story = [
        """Hi Lucy! I'm so glad you just finished reading "All the Beast's Together" (I think that might be a slight mistake, it's actually called "The Jungle Book"?)...""",
        """That was quite a wild and exciting adventure for Mowgli!""",
        """What did you think of Baloo, the friendly bear who became Mowgli's friend?...Did he make you laugh or smile at all in the story?" """,
    ]

    async def receiver():
        speaker = Speaker()
        speaker.start()
        try:
            while True:
                if _socket is None or _exit.is_set():
                    break

                message = _socket.recv()
                if message is None:
                    continue

                if type(message) is str:
                    print(message)
                elif type(message) is bytes:
                    speaker.play(message)
        except Exception as e:
            print(f"receiver: {e}")
        finally:
            speaker.stop()

    _receiver_thread = threading.Thread(target=asyncio.run, args=(receiver(),))
    _receiver_thread.start()

    for text_input in _story:
        print(f"Sending: {text_input}")
        _socket.send(json.dumps({"type": "Speak", "text": text_input}))

    print("Flushing...")
    _socket.send(json.dumps({"type": "Flush"}))

    input("Press Enter to exit...")
    _exit.set()
    _socket.send(json.dumps({"type": "Close"}))
    _socket.close()

    _listen_thread.join()
    _listen_thread = None


class Speaker:
    _audio: pyaudio.PyAudio
    _chunk: int
    _rate: int
    _format: int
    _channels: int
    _output_device_index: int

    _stream: pyaudio.Stream
    _thread: threading.Thread
    _asyncio_loop: asyncio.AbstractEventLoop
    _asyncio_thread: threading.Thread
    _queue: queue.Queue
    _exit: threading.Event

    def __init__(
        self,
        rate: int = RATE,
        chunk: int = CHUNK,
        channels: int = CHANNELS,
        output_device_index: int = None,
    ):
        self._exit = threading.Event()
        self._queue = queue.Queue()

        self._audio = pyaudio.PyAudio()
        self._chunk = chunk
        self._rate = rate
        self._format = FORMAT
        self._channels = channels
        self._output_device_index = output_device_index

    def _start_asyncio_loop(self) -> None:
        self._asyncio_loop = asyncio.new_event_loop()
        self._asyncio_loop.run_forever()

    def start(self) -> bool:
        self._stream = self._audio.open(
            format=self._format,
            channels=self._channels,
            rate=self._rate,
            input=False,
            output=True,
            frames_per_buffer=self._chunk,
            output_device_index=self._output_device_index,
        )

        self._exit.clear()

        self._thread = threading.Thread(
            target=_play, args=(self._queue, self._stream, self._exit), daemon=True
        )
        self._thread.start()

        self._stream.start_stream()

        return True

    def stop(self):
        self._exit.set()

        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        self._thread.join()
        self._thread = None

        self._queue = None

    def play(self, data):
        self._queue.put(data)


def _play(audio_out: queue, stream, stop):
    while not stop.is_set():
        try:
            data = audio_out.get(True, TIMEOUT)
            stream.write(data)
        except queue.Empty as e:
            # print(f"queue is empty")
            pass
        except Exception as e:
            print(f"_play: {e}")

if __name__ == "__main__":
    main()
