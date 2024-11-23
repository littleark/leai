import speech_recognition as sr

r = sr.Recognizer()

def listen_for_long_break():
    full_text = ""  # To accumulate the recognized text
    with sr.Microphone() as source:
        print("Listening for speech (will stop after a long break)...")
        r.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        while True:
            try:
                # Listen with a maximum duration of 5 seconds and wait for a short break of 1 second
                audio_data = r.listen(source, timeout=5, phrase_time_limit=5)
                print("Recognizing...")

                # Convert speech to text using Google Web Speech API
                text = r.recognize_google(audio_data)
                print(f"You said: {text}")

                # Accumulate the recognized text
                full_text += text + " "

            except sr.UnknownValueError:
                print("Sorry, I could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
            except sr.WaitTimeoutError:
                # If no speech detected for 5 seconds, break the loop
                print("No speech detected for a long time, stopping.")
                break  # Exit the loop on long silence (timeout)

    return full_text.strip()  # Return the accumulated text (remove trailing whitespace)

# Start listening and get the full text
# full_text = listen_for_long_break()
# print(f"Full recognized text: {full_text}")
