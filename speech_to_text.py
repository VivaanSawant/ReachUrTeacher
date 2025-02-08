import speech_recognition as sr
import time
import threading
from datetime import datetime

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise, please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Helps with background noise
        recognizer.pause_threshold = 1  # Allows short pauses without breaking recognition
        print("Listening continuously... (Press Ctrl+C to stop)")
        captured_text = []
        lock = threading.Lock()  # Prevents race conditions when updating captured_text

        def listen_continuously():
            with sr.Microphone() as mic_source:
                while True:
                    try:
                        recognizer.adjust_for_ambient_noise(mic_source, duration=0.5)  # Constantly adjust
                        audio = recognizer.listen(mic_source)  # Continuous listening
                        text = recognizer.recognize_google(audio)
                        with lock:
                            captured_text.append(text)
                    except sr.UnknownValueError:
                        pass  # Ignore unintelligible segments instead of appending junk
                    except sr.RequestError:
                        with lock:
                            captured_text.append("[API Error]")

        def print_periodically():
            while True:
                time.sleep(5)
                with lock:
                    if captured_text:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[{timestamp}]: {' '.join(captured_text)}")
                        captured_text.clear()
                    else:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[{timestamp}] No speech detected in the last 5 seconds.")

        # Start both threads
        listening_thread = threading.Thread(target=listen_continuously, daemon=True)
        print_thread = threading.Thread(target=print_periodically, daemon=True)
        listening_thread.start()
        print_thread.start()

        listening_thread.join()
        print_thread.join()

if __name__ == "__main__":
    try:
        recognize_speech()
    except KeyboardInterrupt:
        print("\nStopping speech recognition.")
