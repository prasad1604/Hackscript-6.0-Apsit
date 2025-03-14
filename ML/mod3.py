import librosa
import soundfile as sf
import speech_recognition as sr
from transformers import pipeline
from pyannote.audio import Pipeline
import time
import os
# Pre-trained models
sentiment_pipeline = pipeline('sentiment-analysis')
speaker_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token="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"
)
recognizer = sr.Recognizer()

# File path
file_path = "./datasets/audio.wav"

# Function to perform real-time-like processing on the audio file in chunks
def process_audio_in_chunks(file_path, chunk_duration=5):
    y, sample_rate = librosa.load(file_path, sr=None)

    # Get the total duration of the audio
    total_duration = librosa.get_duration(y=y, sr=sample_rate)
    

    # Initialize variables for sentiment and speaker diarization
    last_customer_end = 0
    response_times = []

    # Loop through audio in chunks
    for start in range(0, int(total_duration), chunk_duration):
        end = min(start + chunk_duration, total_duration)
        audio_chunk = y[int(start * sample_rate):int(end * sample_rate)]
        audio_chunk_path = f"./datasets/audio_chunk_{start}_{end}.wav"
        
        # Save the chunk to a file for processing
        sf.write(audio_chunk_path, audio_chunk, sample_rate)

        # Perform speaker diarization
        diarization = speaker_pipeline(audio_chunk_path)
        
        customer_audio = []
        agent_audio = []

        # Split audio based on diarization
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker == "SPEAKER_00":  # Customer
                customer_audio.extend(audio_chunk[int(turn.start * sample_rate):int(turn.end * sample_rate)])
                last_customer_end = turn.end
            elif speaker == "SPEAKER_01":  # Agent
                agent_audio.extend(audio_chunk[int(turn.start * sample_rate):int(turn.end * sample_rate)])
                response_time = turn.start - last_customer_end
                if response_time > 0:
                    response_times.append(response_time)

        # Save the split customer and agent audio
        customer_path = "./datasets/customer_chunk.wav"
        agent_path = "./datasets/agent_chunk.wav"

        sf.write(customer_path, customer_audio, sample_rate)
        sf.write(agent_path, agent_audio, sample_rate)

        # Perform speech recognition and sentiment analysis for customer
        try:
            with sr.AudioFile(customer_path) as source:
                audio = recognizer.record(source)
                customer_text = recognizer.recognize_google(audio)
                customer_sentiment = sentiment_pipeline(customer_text)[0]['label']
                print(f"Customer: {customer_text} | Sentiment: {customer_sentiment}")
        except Exception as e:
            print(f"Error processing customer speech in chunk {start}-{end}: {e}")

        # Perform speech recognition and sentiment analysis for agent
        try:
            with sr.AudioFile(agent_path) as source:
                audio = recognizer.record(source)
                agent_text = recognizer.recognize_google(audio)
                agent_sentiment = sentiment_pipeline(agent_text)[0]['label']
                print(f"Agent: {agent_text} | Sentiment: {agent_sentiment}")
        except Exception as e:
            print(f"Error processing agent speech in chunk {start}-{end}: {e}")

        # Clean up the chunk files
        os.remove(audio_chunk_path)
        os.remove(customer_path)
        os.remove(agent_path)

    # Calculate average response time
    if response_times:
        avg_response_time = sum(response_times) / len(response_times)
        print(f"Average Response Time: {avg_response_time:.2f} seconds")

if __name__ == '__main__':
    process_audio_in_chunks(file_path)
