import speech_recognition as sr
import time
from datetime import datetime

speech_data = []

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        while True:
            try:
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                speech_data.append({"timestamp": timestamp, "text": text})
                print(f"[{timestamp}] {text}")
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                speech_data.append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "text": "[API Error]"})

if __name__ == "__main__":
    recognize_speech()
