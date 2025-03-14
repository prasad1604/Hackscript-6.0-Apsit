from flask import Flask, request, jsonify, render_template
import os
import librosa
import speech_recognition as sr
from textblob import TextBlob
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
current_path = os.getcwd()+"/datasets"
print("this is my path:", current_path)

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')
recognizer = sr.Recognizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    # Get uploaded audio files
    customer_audio = request.files['customer_audio']
    agent_audio = request.files['agent_audio']

    # Save audio files temporarily
    customer_path = os.path.join(current_path, "customer.wav")
    agent_path = os.path.join(current_path, "agent.wav")

    customer_audio.save(customer_path)
    agent_audio.save(agent_path)

    # Load audio files
    y, sr2 = librosa.load(customer_path, sr=None)
    y1, sr1 = librosa.load(agent_path, sr=None)

    # Convert speech to text
    with sr.AudioFile(customer_path) as source:
        audio = recognizer.record(source)
        customer_text = recognizer.recognize_google(audio)

    with sr.AudioFile(agent_path) as source1:
        audio1 = recognizer.record(source1)
        agent_text = recognizer.recognize_google(audio1)

    # Sentiment Analysis
    customer_sentiment = sentiment_pipeline(customer_text)[0]
    agent_sentiment = sentiment_pipeline(agent_text)[0]

    # Keyword Extraction
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit_transform([customer_text])
    customer_keywords = vectorizer.get_feature_names_out()

    vectorizer.fit_transform([agent_text])
    agent_keywords = vectorizer.get_feature_names_out()

    # Response Data
    response = {
        "customer_text": customer_text,
        "agent_text": agent_text,
        "customer_sentiment": customer_sentiment,
        "agent_sentiment": agent_sentiment,
        "customer_keywords": customer_keywords.tolist(),
        "agent_keywords": agent_keywords.tolist()
    }

    # Alert for negative sentiment
    if customer_sentiment['label'] == 'NEGATIVE':
        response['customer_alert'] = "🚨 Negative Sentiment Detected from Customer!"

    if agent_sentiment['label'] == 'NEGATIVE':
        response['agent_alert'] = "🚨 Negative Sentiment Detected from Agent!"

    # Clean up temp files
    os.remove(customer_path)
    os.remove(agent_path)

    return jsonify(response)

if __name__ == '__main__':
    os.makedirs("datasets", exist_ok=True)
    app.run(debug=True)
