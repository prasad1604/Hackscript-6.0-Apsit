import os
import time
import numpy as np
import librosa
import soundfile as sf
import speech_recognition as sr
from transformers import pipeline
from pyannote.audio import Pipeline as SpeakerPipeline
from flask import Flask, render_template, request

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), "datasets")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Instantiate pipelines:
# Sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

# Semantic analysis pipeline using your fine-tuned model (assumed saved in "./fine_tuned_mnli_quick")
semantic_pipeline = pipeline("zero-shot-classification", model="./fine_tuned_mnli_quick")
if not semantic_pipeline.model.config.label2id:
    semantic_pipeline.model.config.label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}

# Mapping from MultiNLI labels to simplified relevance labels.
label_mapping = {
    "entailment": "relevant",
    "contradiction": "irrelevant",
    "neutral": "neutral"
}

# Use pyannote.audio for diarization.
speaker_pipeline = SpeakerPipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token="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"
)

# Configure the speech recognizer.
recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 0.8

def enhance_audio(audio_samples):
    """
    Apply a pre-emphasis filter to boost high frequencies.
    This can help improve speech recognition for low-energy segments.
    """
    audio_array = np.array(audio_samples)
    enhanced = librosa.effects.preemphasis(audio_array)
    return enhanced.tolist()

def process_audio_real_time(file_path, chunk_duration=15):
    results = []
    try:
        y, sample_rate = librosa.load(file_path, sr=None)
    except Exception as e:
        results.append(f"Failed to load audio: {repr(e)}")
        return results

    total_duration = librosa.get_duration(y=y, sr=sample_rate)
    results.append(f"Total duration: {total_duration:.2f}s")
    response_times = []
    
    # Dictionary to collect segments per speaker.
    # Each segment is stored as (absolute_start, absolute_end, audio_segment)
    speaker_segments = {}

    for start in range(0, int(total_duration), chunk_duration):
        end = min(start + chunk_duration, total_duration)
        audio_chunk = y[int(start * sample_rate):int(end * sample_rate)]
        chunk_path = os.path.join(app.config['UPLOAD_FOLDER'], f"audio_chunk_{start}_{end}.wav")
        
        try:
            sf.write(chunk_path, audio_chunk, sample_rate)
            diarization = speaker_pipeline(chunk_path)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                abs_start = start + turn.start
                abs_end = start + turn.end
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                start_idx = int(turn.start * sample_rate)
                end_idx = int(turn.end * sample_rate)
                segment = audio_chunk[start_idx:end_idx]
                speaker_segments[speaker].append((abs_start, abs_end, segment))
            os.remove(chunk_path)
            results.append(f"Processed chunk {start}-{end}s.")
        except Exception as e:
            results.append(f"Chunk {start}-{end}s error: {str(e)}")
            continue

    # Count long pauses among agent segments.
    agent_times = []
    for spk, segs in speaker_segments.items():
        if spk == "SPEAKER_01":
            for seg in segs:
                agent_times.append((seg[0], seg[1]))
    long_pause_flags = 0
    if len(agent_times) > 1:
        agent_times.sort(key=lambda x: x[0])
        for i in range(1, len(agent_times)):
            gap = agent_times[i][0] - agent_times[i-1][1]
            if gap > 3:
                long_pause_flags += 1
                results.append(f"Long pause: {gap:.2f}s between segments {i-1} and {i}")
    results.append(f"Long pauses (>3s): {long_pause_flags}")
    
    # Heuristic for agent selection.
    if not speaker_segments:
        results.append("No speaker segments detected.")
        return results
    elif len(speaker_segments) == 1:
        agent_label = list(speaker_segments.keys())[0]
        results.append(f"Only one speaker detected ({agent_label}). Assuming this is the agent.")
    else:
        if "SPEAKER_01" in speaker_segments:
            agent_label = "SPEAKER_01"
            results.append("Multiple speakers detected. Using SPEAKER_01 as agent (heuristic).")
        else:
            total_durations = {spk: sum([seg[1] - seg[0] for seg in segs]) for spk, segs in speaker_segments.items()}
            agent_label = min(total_durations, key=total_durations.get)
            results.append(f"Multiple speakers detected. Choosing {agent_label} as agent (shorter total duration).")
    results.append(f"Agent label selected: {agent_label}")
    
    # Calculate response times between customer and agent segments.
    customer_segments = []
    for spk, segs in speaker_segments.items():
        if spk != agent_label:
            customer_segments.extend(segs)
    customer_segments.sort(key=lambda x: x[1])
    for cust in customer_segments:
        cust_end = cust[1]
        for agent_seg in speaker_segments[agent_label]:
            if agent_seg[0] > cust_end:
                response_times.append(agent_seg[0] - cust_end)
                break
    if response_times:
        avg_response = sum(response_times) / len(response_times)
        results.append(f"Avg response time: {avg_response:.2f}s")
    else:
        results.append("No response times available")
    
    # Merge all agent segments.
    try:
        merged_agent_audio = np.concatenate([seg[2] for seg in speaker_segments[agent_label]])
        merged_agent_audio = np.array(enhance_audio(merged_agent_audio.tolist()))
        aggregated_agent_path = os.path.join(app.config['UPLOAD_FOLDER'], "agent_full.wav")
        sf.write(aggregated_agent_path, merged_agent_audio, sample_rate)
        agg_duration = librosa.get_duration(y=merged_agent_audio, sr=sample_rate)
        results.append(f"Aggregated agent audio created. Duration: {agg_duration:.2f} seconds")
    except Exception as e:
        results.append(f"Error merging agent audio: {str(e)}")
        return results

    # Process aggregated agent audio for speech recognition.
    agent_text = ""
    try:
        with sr.AudioFile(aggregated_agent_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            audio_data = recognizer.record(source)
            agent_text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        results.append("Aggregated Agent Conversation: No clear speech recognized from aggregation.")
    except Exception as e:
        results.append(f"Error processing aggregated agent speech: {str(e)}")
    
    # Fallback: Process individual agent segments if aggregation yields no text.
    if not agent_text:
        results.append("Fallback: processing individual agent segments.")
        agent_texts = []
        for seg in speaker_segments[agent_label]:
            try:
                temp_seg_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_seg.wav")
                sf.write(temp_seg_path, seg[2], sample_rate)
                with sr.AudioFile(temp_seg_path) as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data_seg = recognizer.record(source)
                    agent_texts.append(recognizer.recognize_google(audio_data_seg))
                os.remove(temp_seg_path)
            except Exception as e:
                results.append(f"Error processing an agent segment: {str(e)}")
        if agent_texts:
            agent_text = " ".join(agent_texts)
    
    # Initialize flag counter.
    flag_count = long_pause_flags
    negative_flag = 0
    irrelevant_flag = 0

    if agent_text:
        try:
            sentiment = sentiment_pipeline(agent_text)[0]
            results.append(f"Sentiment: {sentiment['label']} (score: {sentiment['score']:.2f})")
            if sentiment['label'].upper() == "NEGATIVE":
                negative_flag += 1
                results.append("Flag: Negative sentiment detected")
            semantic = semantic_pipeline(agent_text, candidate_labels=["entailment", "neutral", "contradiction"])
            sem_label = label_mapping.get(semantic["labels"][0], "neutral")
            results.append(f"Semantic: {sem_label}")
            if sem_label == "irrelevant":
                irrelevant_flag += 1
                results.append("Flag: Irrelevant content detected")
            flag_count += negative_flag + irrelevant_flag
            results.append(f"Agent text: {agent_text}")
        except Exception as e:
            results.append(f"Error analyzing aggregated agent text: {str(e)}")
    else:
        results.append("Aggregated Agent Conversation: No clear speech recognized after fallback.")
    
    # NEW OVERRIDE: if sentiment is negative or semantic is irrelevant, force performance to VERY BAD.
    if negative_flag > 0 or irrelevant_flag > 0:
        performance = "VERY BAD"
    else:
        if flag_count == 0:
            performance = "EXCELLENT"
        elif flag_count == 1:
            performance = "GOOD"
        elif flag_count <= 3:
            performance = "AVERAGE"
        elif flag_count <= 5:
            performance = "BAD"
        else:
            performance = "VERY BAD"
    results.append(f"Total Flags: {flag_count}")
    results.append(f"Agent Performance Rating: {performance}")
    
    try:
        os.remove(aggregated_agent_path)
    except Exception:
        pass

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return "No file", 400
    file = request.files['audio']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    results = process_audio_real_time(file_path)
    os.remove(file_path)
    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
