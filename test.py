import wave
import numpy as np
import sounddevice as sd
from piper import PiperVoice

voice = PiperVoice.load("en_US-lessac-medium.onnx")

# Save to file
with wave.open("output.wav", "wb") as wav_file:
    voice.synthesize_wav("Hello, this is your assistant speaking.", wav_file)

# Streaming playback
text = "Hey my name is Anis, I am your assistant. I am here to help you with anything you need. Just ask me anything and I will do my best to assist you.   "

for chunk in voice.synthesize(text):
    audio_data = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
    sd.play(audio_data, samplerate=chunk.sample_rate)
    sd.wait()