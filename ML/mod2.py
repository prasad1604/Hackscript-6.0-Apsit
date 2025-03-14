import torch
from pyannote.audio import Pipeline
import os

# Load the pre-trained model from HuggingFace
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token="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"
)

# Path to your audio file
file=os.getcwd()
file_path = "./datasets/audio.wav"

# Run the diarization pipeline
diarization = pipeline(file_path)

# Create two separate audio files for customer and agent
customer_audio = []
agent_audio = []

# Split the audio based on speaker
for turn, _, speaker in diarization.itertracks(yield_label=True):
    if speaker == "SPEAKER_00":
        customer_audio.append((turn.start, turn.end))
    elif speaker == "SPEAKER_01":
        agent_audio.append((turn.start, turn.end))

print("Customer Audio Segments:", customer_audio)
print("Agent Audio Segments:", agent_audio)

# Save audio files
import librosa
import soundfile as sf

# Load the audio file
y, sr = librosa.load(file_path, sr=None)

# Function to save audio segments
def save_audio(segments, output_file):
    audio_data = []
    for start, end in segments:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        audio_data.extend(y[start_sample:end_sample])
    sf.write(output_file, audio_data, sr)

# Save Customer and Agent Audio
save_audio(customer_audio, "datasets/customer.wav")
save_audio(agent_audio, "datasets/agent.wav")

print("✅ Audio separated successfully!")

current_path = os.getcwd()
file_path = os.path.join(current_path, "/datasets/customer.wav")
file_path1 = os.path.join(current_path, "/datasets/agent.wav")

print("File Path:", file_path)

import time

# Example response time calculation

y,sr=librosa.load(file+"/datasets/customer.wav",sr=None)

y1,sr1=librosa.load(file+"/datasets/agent.wav",sr=None)

mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc1=librosa.feature.mfcc(y=y1,sr=sr1,n_mfcc=13)
# print(mfcc,mfcc1)

import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Convert Customer Audio to Text
with sr.AudioFile(file+"/datasets/customer.wav") as source:
    audio = recognizer.record(source)
    customer_text = recognizer.recognize_google(audio)
with sr.AudioFile(file+"/datasets/agent.wav") as source1:
    audio1=recognizer.record(source1)
    call_center_text=recognizer.recognize_google(audio1)

print("Customer: ", customer_text)
print("call center:",call_center_text)


from transformers import pipeline


sentiment_pipeline = pipeline('sentiment-analysis')
customer_sentiment = sentiment_pipeline(customer_text)[0]
agent_sentiment = sentiment_pipeline(call_center_text)[0]
print("Customer Sentiment: ", customer_sentiment)
print("Agent Sentiment: ", agent_sentiment)


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform([customer_text])

keywords = vectorizer.get_feature_names_out()
print("Keywords: ", keywords)

vect=TfidfVectorizer(stop_words="english")
x1=vectorizer.fit_transform([call_center_text])
keywords1=vectorizer.get_feature_names_out()
print("keywords:",keywords1)

if customer_sentiment['label'] == 'NEGATIVE':
    print("🚨 Alert: Negative Sentiment Detected from Customer!")
if agent_sentiment['label'] == 'NEGATIVE':
    print("🚨 Alert: Negative Sentiment Detected from Agent!")