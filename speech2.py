import pyttsx3
engine = pyttsx3.init()

# For Mac, If you face error related to "pyobjc" when running the `init()` method :
# Install 9.0.1 version of pyobjc : "pip install pyobjc>=9.0.1"
voices = engine.getProperty('voices')
for v in voices:
    print(v)

engine.setProperty('voice', 'com.apple.eloquence.en-US.Grandma')

engine.say("I will speak this text")
engine.runAndWait()
