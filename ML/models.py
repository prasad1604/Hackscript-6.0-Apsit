import matplotlib.pyplot as plt
import librosa

import os
from transformers import pipeline

current_path = os.getcwd()
file_path = os.path.join(current_path, "datasets/1.wav")
file_path1 = os.path.join(current_path, "datasets/2.wav")

print("File Path:", file_path)

import time

# Example response time calculation

y,sr=librosa.load(file_path,sr=None)

y1,sr1=librosa.load(file_path1,sr=None)

mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc1=librosa.feature.mfcc(y=y1,sr=sr1,n_mfcc=13)

# print(mfcc)


import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Convert Customer Audio to Text
with sr.AudioFile(file_path) as source:
    audio = recognizer.record(source)
    customer_text = recognizer.recognize_google(audio)
with sr.AudioFile(file_path1) as source1:
    audio1=recognizer.record(source1)
    call_center_text=recognizer.recognize_google(audio1)

print("Customer: ", customer_text)
print("call center:",call_center_text)


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







    
