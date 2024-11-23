import os
from RealtimeTTS import TextToAudioStream, SystemEngine, ElevenlabsEngine, CoquiEngine
from dotenv import load_dotenv

load_dotenv()
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

def main():
    # engine = ElevenlabsEngine(elevenlabs_api_key)
    engine = CoquiEngine();
    stream = TextToAudioStream(engine)

    stream.feed("Hello world! How are you today?")
    stream.play_async()

if __name__ == '__main__':
    main()

# 1:22:50
