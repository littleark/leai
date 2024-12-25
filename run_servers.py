import asyncio
from api import app
import uvicorn
from AudioTranscriptionServer import AudioTranscriptionServer
import threading

async def run_fastapi():
    config = uvicorn.Config(app, host="0.0.0.0", port=7860)
    server = uvicorn.Server(config)
    await server.serve()

async def run_audio_server():
    transcription_server = AudioTranscriptionServer()
    await transcription_server.start_server()

async def main():
    await asyncio.gather(
        run_fastapi(),
        run_audio_server()
    )

if __name__ == "__main__":
    asyncio.run(main())
