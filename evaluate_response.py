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
from job_role_library import job_role_library  # Import job role library
from sector_library import get_company_sector   # Import the function from sector_library

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

class TranscriptionError(Exception):
    """Custom exception for audio transcription errors."""
    pass

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
        # Ensure company_info is a dict
        if isinstance(company_info, str):
            company_info = json.loads(company_info)
        company_info_list = company_info
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

def get_relevant_keywords(question_data: Dict[str, Any], job_role: str, company_name: str, company_info: str) -> Tuple[List[str], str, str]:
    """
    Retrieves relevant keywords and directly pulls the question type from question_data.
    Also extracts the company sector using get_company_sector() from sector_library.
    Returns a tuple: (keywords, question_type, company_sector)
    """
    job_role_lower = job_role.lower()
    matched_keywords = set()

    # Match job role from library
    for domain, content in job_role_library.items():
        if domain.lower() in job_role_lower:
            if isinstance(content, dict):  # Specialized roles
                for role, keywords in content.items():
                    if role.lower() in job_role_lower:
                        matched_keywords.update(keywords)
                if "keywords" in content:
                    matched_keywords.update(content["keywords"])
            elif isinstance(content, list):
                matched_keywords.update(content)

    # Add keywords from question data
    question_skills = question_data.get("skills", [])
    experience = question_data.get("experience", "")
    question_type = question_data.get("question_type", "general").lower()
    matched_keywords.update(question_skills)
    if experience:
        matched_keywords.update(experience.split(","))
    
    # Add company info keywords
    company_keywords = extract_company_keywords(company_info)
    matched_keywords.update(company_keywords)

    # Add context based on question type
    if question_type == "behavioral":
        matched_keywords.update([
            "time management", "task prioritisation", "communication", "collaboration",
            "problem solving", "conflict resolution", "leadership", "accountability",
            "ownership", "initiative", "conflict management"
        ])
    elif question_type == "situational":
        matched_keywords.update([
            "problem solving", "management", "scenario planning", "decision making",
            "evaluate options", "compromise", "impact assessment", "adaptability",
            "flexibility", "proactive", "contingency plan", "stakeholder management",
            "risk management", "implementation", "task prioritisation"
        ])
    
    # Ensure company_info is a dict before passing to get_company_sector
    if isinstance(company_info, str):
        company_info_dict = json.loads(company_info)
    else:
        company_info_dict = company_info
        
    company_sector = get_company_sector(company_info_dict)
    
    return list(set(matched_keywords)), question_type, company_sector

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
    suggestions = [{
        "area": "Scoring Explanation",
        "feedback": (
            f"Your final score is {final_score} out of 10. "
            "A higher score indicates that you effectively used the expected keywords "
            "and aligned your answer with the question's focus."
        )
    }]
    if missing_keywords:
        suggestions.append({
            "area": "Keyword Usage",
            "feedback": (
                "Consider incorporating the following missing keywords for a more comprehensive answer: "
                f"{', '.join(missing_keywords)}."
            )
        })
    if question_type == "behavioral":
        suggestions.append({
            "area": "Behavioral Structure",
            "feedback": "Use the STAR method (Situation, Task, Action, Result) to structure your answer clearly."
        })
    elif question_type == "situational":
        suggestions.append({
            "area": "Situational Strategy",
            "feedback": "Focus on problem-solving steps and logical reasoning in hypothetical scenarios."
        })
    elif question_type == "technical":
        suggestions.append({
            "area": "Technical Depth",
            "feedback": "Include definitions, examples, or best practices to demonstrate thorough technical knowledge."
        })
    if len(response_text.split()) < 20:
        suggestions.append({
            "area": "Detail & Depth",
            "feedback": "Try to expand your answer with specific examples or explanations to provide more insight."
        })
    return suggestions

def score_response(
    response_text: str,
    question_text: str,
    relevant_keywords: List[str],
    question_type: str,
    company_sector: str
) -> Dict[str, Any]:
    try:
        doc = nlp(response_text)
        response_tokens = [
            token.lemma_.lower().strip(string.punctuation)
            for token in doc if not token.is_stop and len(token.text) > 2
        ]
        cleaned_keywords = [
            kw.lower().strip(string.punctuation)
            for kw in relevant_keywords if len(kw.strip()) > 2
        ]
        matched_keywords, missing_keywords = [], []
        for kw in cleaned_keywords:
            best_match = max((fuzz.partial_ratio(kw, token) for token in response_tokens), default=0)
            if best_match >= 80:
                matched_keywords.append(kw)
            else:
                missing_keywords.append(kw)
        keyword_score = (len(matched_keywords) / len(cleaned_keywords)) * 100 if cleaned_keywords else 0
        question_score = fuzz.partial_ratio(response_text, question_text)
        weights = {
            "technical": {"keyword": 0.5, "question": 0.5},
            "behavioral": {"keyword": 0.3, "question": 0.7},
            "situational": {"keyword": 0.3, "question": 0.7},
        }.get(question_type, {"keyword": 0.4, "question": 0.6})
        final_numeric = round((weights["keyword"] * keyword_score) + (weights["question"] * question_score), 2)
        score_out_of_10 = round(final_numeric / 10, 1)
        suggestions = generate_improvement_suggestions(
            response_text=response_text,
            missing_keywords=missing_keywords,
            question_type=question_type,
            relevant_keywords=cleaned_keywords,
            final_score=score_out_of_10,
            keyword_score=keyword_score,
            question_score=question_score,
            weights=weights
        )
        return {
            "score": score_out_of_10,
            "keyword_matches": matched_keywords,
            "expected_keywords": cleaned_keywords,
            "skills_shown": extract_skills_from_response(response_text),
            "improvement_suggestions": suggestions,
            "transcribed_text": response_text,
            "company_sector": company_sector
        }
    except Exception as e:
        logger.exception("Error scoring response")
        return {"score": 0, "error": str(e)}
