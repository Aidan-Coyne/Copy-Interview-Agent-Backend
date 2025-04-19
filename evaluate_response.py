import speech_recognition as sr
import spacy
from fuzzywuzzy import fuzz
import json
import logging
from typing import List, Dict, Tuple, Any
from fastapi import HTTPException
from io import BytesIO
import wave
import subprocess, uuid, os
import string
import tempfile
from google.cloud import storage

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

class TranscriptionError(Exception):
    pass

# ✅ NEW: Download user audio response from Firebase
def download_user_response_from_firebase(session_id: str, question_index: int, bucket: storage.Bucket) -> bytes:
    """
    Downloads the user's recorded response audio from Firebase Storage and returns the raw bytes.
    """
    blob_path = f"sessions/{session_id}/audio_responses/response_{question_index + 1}.mp3"
    blob = bucket.blob(blob_path)

    if not blob.exists():
        logger.error(f"Audio response not found in Firebase: {blob_path}")
        raise HTTPException(status_code=404, detail="Audio response not found in storage.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        blob.download_to_filename(temp_file.name)
        logger.info(f"Downloaded audio response to temp file: {temp_file.name}")

    with open(temp_file.name, "rb") as f:
        audio_bytes = f.read()

    try:
        os.remove(temp_file.name)
    except Exception as e:
        logger.warning(f"Failed to delete temp audio file: {e}")

    return audio_bytes

def convert_to_wav(audio_data: bytes) -> bytes:
    input_filename = f"temp_{uuid.uuid4().hex}.webm"
    output_filename = f"temp_{uuid.uuid4().hex}.wav"
    try:
        with open(input_filename, "wb") as f:
            f.write(audio_data)

        subprocess.run([
            "ffmpeg", "-y", "-i", input_filename,
            "-acodec", "pcm_s16le", output_filename
        ], check=True)

        with open(output_filename, "rb") as f:
            wav_data = f.read()
    finally:
        if os.path.exists(input_filename):
            os.remove(input_filename)
        if os.path.exists(output_filename):
            os.remove(output_filename)

    return wav_data

def transcribe_audio(audio_data: bytes) -> str:
    recognizer = sr.Recognizer()
    try:
        with wave.open(BytesIO(audio_data), 'rb') as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            n_channels = wf.getnchannels()
            logger.info(f"Audio properties: sample_rate={sample_rate}, sample_width={sample_width}, channels={n_channels}")

        audio_data_sr = sr.AudioData(audio_data, sample_rate=sample_rate, sample_width=sample_width)
        text = recognizer.recognize_google(audio_data_sr)
        return text.lower()

    except sr.UnknownValueError:
        raise TranscriptionError("Could not understand audio. Please try speaking clearly.")
    except sr.RequestError as e:
        raise TranscriptionError(f"Speech recognition service error: {e}")
    except Exception as e:
        raise TranscriptionError(f"Error transcribing audio: {e}")

def extract_company_keywords(company_info: str) -> List[str]:
    try:
        company_info_list = json.loads(company_info)
        company_keywords = []

        for item in company_info_list:
            snippet = item.get("snippet", "").lower()
            doc = nlp(snippet)
            company_keywords.extend(token.lemma_ for token in doc if token.pos_ in {"NOUN", "VERB"})
            company_keywords.extend(ent.text.lower() for ent in doc.ents)

        return list(set(company_keywords))

    except Exception as e:
        logger.error(f"Error extracting company keywords: {e}")
        return []

def extract_skills_from_response(response_text: str) -> List[str]:
    doc = nlp(response_text)
    return list(set(token.text.lower() for token in doc if token.pos_ in {"NOUN", "VERB"}))

def generate_improvement_suggestions(
    response_text: str,
    missing_keywords: List[str],
    question_type: str,
    relevant_keywords: List[str],
    final_score: float,
    keyword_score: float,
    question_score: float,
    weights: Dict[str, float]
) -> List[Dict[str, str]]:
    suggestions = []
    suggestions.append({
        "area": "Scoring Explanation",
        "feedback": (
            f"Your final score is {final_score} out of 10. This was calculated using: "
            f"(Keyword Score: {keyword_score:.1f}% * {weights.get('keyword', 0)} + "
            f"Question Relevance Score: {question_score:.1f} * {weights.get('question', 0)}) = {final_score * 10:.1f}% overall."
        )
    })

    if missing_keywords:
        suggestions.append({
            "area": "Keyword Usage",
            "feedback": f"Consider including these missing keywords: {', '.join(missing_keywords)}."
        })

    if question_type == "behavioral":
        suggestions.append({"area": "Behavioral Structure", "feedback": "Use the STAR method (Situation, Task, Action, Result)."})
    elif question_type == "situational":
        suggestions.append({"area": "Situational Strategy", "feedback": "Explain your reasoning and problem-solving steps."})
    elif question_type == "technical":
        suggestions.append({"area": "Technical Depth", "feedback": "Add specific tools, processes, or examples where possible."})

    if len(response_text.split()) < 20:
        suggestions.append({
            "area": "Detail & Depth",
            "feedback": "Try expanding your answer with more examples or explanations."
        })

    return suggestions

def get_relevant_keywords(question_data: Dict[str, Any], job_role: str, company_name: str, company_info: str) -> Tuple[List[str], str]:
    question_skills = question_data.get("skills", [])
    question_experience = question_data.get("experience", "")
    question_text = question_data.get("question_text", "").lower()

    company_keywords = extract_company_keywords(company_info)
    job_role_lemma = nlp(job_role)[0].lemma_ if job_role else ""
    company_name_lemma = nlp(company_name)[0].lemma_ if company_name else ""

    experience_doc = nlp(question_experience)
    experience_lemma = experience_doc[0].lemma_ if len(experience_doc) > 0 else ""

    all_keywords = set(
        [nlp(skill)[0].lemma_.lower() for skill in question_skills if skill] +
        ([experience_lemma.lower()] if experience_lemma else []) +
        company_keywords +
        [job_role_lemma.lower(), company_name_lemma.lower()]
    )

    question_type = "general"
    if "tell me about a time" in question_text or "describe a situation" in question_text:
        question_type = "behavioral"
    elif "how would you" in question_text:
        question_type = "situational"
    elif "skills" in question_text or "explain" in question_text:
        question_type = "technical"

    return list(all_keywords), question_type

def score_response(
    response_text: str,
    question_text: str,
    relevant_keywords: List[str],
    question_type: str
) -> Dict[str, Any]:
    doc = nlp(response_text)

    response_tokens = [token.lemma_.lower().strip(string.punctuation)
                       for token in doc if not token.is_stop and len(token.text) > 2]

    cleaned_keywords = [kw.lower().strip(string.punctuation) for kw in relevant_keywords if len(kw.strip()) > 2]

    matched_keywords = []
    missing_keywords = []
    threshold = 80

    for kw in cleaned_keywords:
        best_ratio = max(fuzz.partial_ratio(kw, token) for token in response_tokens)
        if best_ratio >= threshold:
            matched_keywords.append(kw)
        else:
            missing_keywords.append(kw)

    keyword_score = (len(matched_keywords) / len(cleaned_keywords)) * 100 if cleaned_keywords else 0
    question_score = fuzz.partial_ratio(response_text, question_text)

    if question_type == "technical":
        weights = {"keyword": 0.5, "question": 0.5}
    elif question_type == "behavioral":
        weights = {"keyword": 0.3, "question": 0.7}
    elif question_type == "situational":
        weights = {"keyword": 0.3, "question": 0.7}
    else:
        weights = {"keyword": 0.4, "question": 0.6}

    final_numeric = round((weights["keyword"] * keyword_score) + (weights["question"] * question_score), 2)
    score_out_of_10 = round(final_numeric / 10, 1)

    suggestions = generate_improvement_suggestions(
        response_text, missing_keywords, question_type, cleaned_keywords,
        score_out_of_10, keyword_score, question_score, weights
    )

    return {
        "score": score_out_of_10,
        "keyword_matches": matched_keywords,
        "expected_keywords": cleaned_keywords,
        "skills_shown": extract_skills_from_response(response_text),
        "improvement_suggestions": suggestions,
        "transcribed_text": response_text
    }
