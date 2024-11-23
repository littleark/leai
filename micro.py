import speech_recognition as sr
import threading
import queue

def listen(timeout=30, phrase_time_limit=5, stop_keyword="full stop"):
    """
    Listen to microphone input and return transcribed text.

    Args:
        timeout (int): Total time to listen for speech, defaults to 30 seconds
        phrase_time_limit (int): Maximum time for each speech segment, defaults to 5 seconds
        stop_keyword (str): Keyword to stop listening, defaults to "full stop"

    Returns:
        str: Transcribed text from microphone input
    """
    # Create recognizer and microphone instances
    recognizer = sr.Recognizer()

    # Text accumulator and synchronization queue
    full_text = ""
    stop_queue = queue.Queue()

    def listening_thread():
        nonlocal full_text
        with sr.Microphone() as source:
            print("Listening for speech...")
            recognizer.adjust_for_ambient_noise(source)

            try:
                while True:
                    try:
                        # Listen with timeout and phrase limit
                        audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

                        # Transcribe audio
                        text = recognizer.recognize_google(audio_data)
                        print(f"You said: {text}")

                        # Check for stop keyword
                        if stop_keyword.lower() in text.lower():
                            print(f"Keyword '{stop_keyword}' detected. Stopping.")
                            stop_queue.put(True)
                            break

                        # Accumulate text
                        full_text += text + " "

                    except sr.UnknownValueError:
                        print("Could not understand audio.")
                    except sr.WaitTimeoutError:
                        print("No speech detected, stopping.")
                        stop_queue.put(True)
                        break

            except Exception as e:
                print(f"An error occurred: {e}")
                stop_queue.put(True)

    # Start listening in a thread
    listener_thread = threading.Thread(target=listening_thread)
    listener_thread.start()

    # Wait for the thread to complete or be stopped
    stop_queue.get()
    listener_thread.join()

    return full_text.strip()

# Example usage if run directly
if __name__ == "__main__":
    try:
        transcribed_text = listen()
        print(f"Final transcribed text: {transcribed_text}")
    except Exception as e:
        print(f"Error in speech recognition: {e}")
