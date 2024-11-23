import multiprocessing
multiprocessing.set_start_method("fork")  # Set start method to fork

from flask import Flask, request, jsonify
from RealtimeTTS import TextToAudioStream, CoquiEngine
from dotenv import load_dotenv
import os

# Load API key if needed
load_dotenv()
app = Flask(__name__)

# Initialize TTS Engine and Stream
tts_engine = CoquiEngine()
tts_stream = TextToAudioStream(tts_engine)

@app.route('/speak', methods=['POST'])
def speak():
    data = request.get_json()
    text = data.get('text', '')
    try:
        tts_stream.feed(text)
        tts_stream.play()
        return jsonify({"status": "success", "message": "Playing audio"}), 200
    except EOFError as e:
        print("EOFError encountered in TTS engine, restarting engine.")
        initialize_tts()  # Reinitialize TTS on failure
        return jsonify({"status": "error", "message": "TTS engine restarted due to error."}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
