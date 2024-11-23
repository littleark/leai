import pyaudio

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("Recording... Press Ctrl+C to stop.")
try:
    while True:
        data = stream.read(CHUNK)
        print(f"Captured {len(data)} bytes")
except KeyboardInterrupt:
    print("Stopped.")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
