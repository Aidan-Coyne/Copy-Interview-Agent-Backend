import os
import json
import wave
import string
import logging
import subprocess
import re
import tempfile
from io import BytesIO
from typing import List, Dict, Tuple, Any, Optional

import spacy
from fuzzywuzzy import fuzz
import onnxruntime as ort
import numpy as np
import webrtcvad
from faster_whisper import WhisperModel
from transformers import AutoTokenizer
from openai import OpenAI

from job_role_library import job_role_library
from app_cache import get_cached_prompt, cache_prompt_to_firestore, get_question_hash
from usage_logger import log_usage_to_supabase

QUESTION_TYPE_GUIDANCE = {
    "technical": (
        "For technical questions, focus on structure, clarity, and specificity. "
        "Explain your process, tools used, and outcomes. Mention challenges and how you resolved them."
    ),
    "behavioral": (
        "For behavioral questions, use the STAR method (Situation, Task, Action, Result). "
        "Share a specific past experience and highlight soft skills like communication and teamwork."
    ),
    "situational": (
        "For situational questions, consider using the CARE method (Context, Action, Reasoning, Evaluation). "
        "Describe how you'd approach the scenario, justify your decisions, and explain how you'd assess the outcome."
    ),
    "general": (
        "For general questions, speak confidently and stay relevant. Reflect on your values, experience, and growth without going off-topic."
    )
}

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

openai_client = OpenAI()
whisper_model = WhisperModel("tiny", compute_type="int8")
logger.info("Loaded FasterWhisper model: tiny")

MODEL_PATH = "/app/models/paraphrase-MiniLM-L3-v2.onnx"
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")
session = ort.InferenceSession(MODEL_PATH)

nlp = spacy.load("en_core_web_sm")

FILLER_WORDS = {
    "um", "uh", "like", "you know", "i mean", "basically", "sort of", "kind of", "er", "erm"
}

class TranscriptionError(Exception):
    pass

def convert_to_wav(audio_data: bytes) -> bytes:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-af", "highpass=f=200, lowpass=f=3000",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"],
        input=audio_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True
    )
    return proc.stdout

def transcribe_audio(audio_data: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            temp_wav.write(audio_data)
            temp_wav_path = temp_wav.name
        segments, _ = whisper_model.transcribe(temp_wav_path, beam_size=1)
        text = " ".join(segment.text for segment in segments).strip()
        if not text:
            raise TranscriptionError("No speech detected.")
        return text.lower()
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}")

def count_filler_words(text: str) -> int:
    words = re.sub(r"[^a-zA-Z\s]", "", text.lower()).split()
    return sum(1 for word in words if word in FILLER_WORDS)

def extract_skills_from_response(text: str) -> List[str]:
    return list({t.text.lower() for t in nlp(text) if t.pos_ in {"NOUN", "VERB"}})

def get_relevant_keywords(question_data: Dict[str, Any], job_role: str, company_name: str, company_info: dict) -> Tuple[List[str], str, Optional[str]]:
    qtext = question_data.get("question_text", "").lower()
    keywords = set()
    matched_sector = None

    job_role_lower = job_role.lower()
    for discipline, roles in job_role_library.items():
        for role_title, kws in roles.items():
            if isinstance(kws, list) and role_title.lower() == job_role_lower:
                keywords.update(kws)
                matched_sector = discipline
                break

    # Determine question type
    stored_qtype = question_data.get("question_type", "").lower()
    if stored_qtype in {"technical", "behavioral", "situational"}:
     qtype = stored_qtype
    else:
    
    # fallback logic if not present or invalid
        if "tell me about a time" in qtext or "describe a situation" in qtext:
            qtype = "behavioral"
        elif "how would you" in qtext:
            qtype = "situational"
        elif "skills" in qtext or "explain" in qtext:
            qtype = "technical"
        else:
            qtype = "general"


    if not keywords:
        logging.warning(f"⚠️ No keywords found for role: {job_role}")
        return [], qtype, None

    lemmatized_keywords = {
        nlp(k)[0].lemma_.lower() for k in keywords if isinstance(k, str) and len(k.strip()) > 2
    }

    return list(lemmatized_keywords), qtype, matched_sector

def question_terms_score(q: str, r: str) -> float:
    q_tokens = {t.lemma_.lower() for t in nlp(q) if not t.is_stop and t.is_alpha and len(t.text) > 2}
    r_lemmas = {t.lemma_.lower() for t in nlp(r)}
    if not q_tokens:
        return 0.0
    matches = sum(1 for tok in q_tokens if tok in r_lemmas)
    return matches / len(q_tokens) * 100

def encode(text: str) -> np.ndarray:
    tokens = tokenizer(text, return_tensors="np", truncation=True, padding="max_length", max_length=128)
    ort_inputs = {k: tokens[k] for k in ("input_ids", "attention_mask")}
    outputs = session.run(None, ort_inputs)
    last_hidden = outputs[0]
    mask = tokens["attention_mask"][..., None]
    return ((last_hidden * mask).sum(axis=1) / mask.sum(axis=1))[0]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def semantic_similarity(q: str, r: str) -> float:
    return ((cosine_similarity(encode(q), encode(r)) + 1) / 2) * 100

def detect_long_pauses(wav_bytes: bytes) -> float:
    vad = webrtcvad.Vad(2)
    try:
        wf = wave.open(BytesIO(wav_bytes), 'rb')
        if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            return 0.0
    except:
        return 0.0
    frame_ms = 30
    frames, speech = 0, 0
    while True:
        data = wf.readframes(int(wf.getframerate() * frame_ms / 1000))
        if not data:
            break
        if vad.is_speech(data, wf.getframerate()):
            speech += 1
        frames += 1
    return 1 - (speech / frames) if frames else 0.0

def clarity_score(r: str, wav_bytes: bytes) -> float:
    pause_penalty = detect_long_pauses(wav_bytes)
    filler_penalty = min(0.05 * count_filler_words(r), 0.25)
    return 100 * (1 - pause_penalty) * (1 - filler_penalty)

def deep_round(value: float, ndigits: int) -> float:
    return round(value, ndigits)

def generate_openai_feedback(question: str, answer: str, q_type: str = "unspecified", user_id: Optional[str] = None) -> List[str]:
    try:
        cached_messages = get_cached_prompt(question)
        if not cached_messages:
            cached_messages = [
                {"role": "system", "content": "You are an AI career coach providing structured interview feedback."},
                {"role": "user", "content": f"""You are evaluating a candidate's interview answer.

            Question:
            \"{question}\"

            Answer:
            \"{{ANSWER_PLACEHOLDER}}\"

            Provide feedback in two clear paragraphs in max 80 tokens:
            1. What they did well (quote from answer).
            2. What could be improved (clarify, expand, structure).
            """}
            ]
            cache_prompt_to_firestore(question, cached_messages, q_type)

        final_messages = [
            msg if msg["role"] == "system" else {
                "role": "user",
                "content": msg["content"].replace("{{ANSWER_PLACEHOLDER}}", answer)
            }
            for msg in cached_messages
        ]

        result = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=final_messages,
            temperature=0.7,
            max_tokens=80,
        )

        if user_id:
            usage = result.usage
            total_tokens = usage.prompt_tokens + usage.completion_tokens
            log_usage_to_supabase(user_id, tokens_used=total_tokens, request_type="generate_openai_feedback")

        return result.choices[0].message.content.strip().split("\n\n")

    except Exception as e:
        logger.error(f"OpenAI feedback failed: {e}")
        return ["(LLM Feedback error)"]

def get_tier(score: float) -> str:
    return "needs_improvement" if score < 40 else "on_track" if score < 70 else "strong"

def score_response(response_text: str, question_text: str, relevant_keywords: List[str], question_type: str, company_sector: Optional[str] = None, user_id: Optional[str] = None, wav_bytes: bytes = b"") -> Dict[str, Any]:
    logger.info("Starting evaluation pipeline")
    tokens = [t.lemma_.lower().strip(string.punctuation) for t in nlp(response_text) if not t.is_stop and len(t.text) > 2]
    kws = [kw.lower().strip(string.punctuation) for kw in relevant_keywords]
    matched, missing = [], []
    for kw in kws:
        pct = max(fuzz.partial_ratio(kw, tok) for tok in tokens) if tokens else 0
        (matched if pct >= 80 else missing).append(kw)
    keyword_score = len(matched) / len(kws) * 100 if kws else 0
    sem = semantic_similarity(question_text, response_text)
    clr = clarity_score(response_text, wav_bytes)
    qterms = question_terms_score(question_text, response_text)
    rel_w = {"semantic": .70, "clarity": .15, "qterms": .15}
    question_score = sem * rel_w["semantic"] + clr * rel_w["clarity"] + qterms * rel_w["qterms"]
    kw_w, qt_w = {"technical": (.4, .6), "behavioral": (.2, .8), "situational": (.2, .8)}.get(question_type, (0.2, 0.8))
    final_num = deep_round(kw_w * keyword_score + qt_w * question_score, 2)
    score10 = round(final_num / 10, 1)

    suggestions = [
        {"area": "Scoring Explanation", "feedback": f"Final score {score10}/10 (Keywords {keyword_score:.1f}%×{kw_w}, Relevance {question_score:.1f}%×{qt_w})."},
        {"area": "Relevance Breakdown", "feedback": f"Topic Fit: {sem:.1f}%  •  Clarity: {clr:.1f}%  •  Question Terms Used: {qterms:.1f}%"}
    ]

    if question_type in QUESTION_TYPE_GUIDANCE:
         suggestions.append({
            "area": "Answering Strategy",
            "feedback": QUESTION_TYPE_GUIDANCE[question_type]
        })


    if sem < 70: suggestions.append({"area": "Topic Fit", "feedback": "Consider aligning your answer more closely with the main topic to improve relevance."})
    if clr < 70: suggestions.append({"area": "Clear Answer", "feedback": "Try to reduce long pauses and avoid filler words."})
    if qterms < 70: suggestions.append({"area": "Question Terms Used", "feedback": "Use key terms from the question to demonstrate understanding."})
    if count_filler_words(response_text) > 0: suggestions.append({"area": "Clear Answer", "feedback": "Reduce filler words like 'uh', 'like', or 'you know'."})
    if missing and relevant_keywords: suggestions.append({"area": "Keyword Usage", "feedback": f"Consider adding missing keywords: {', '.join(missing)}."})
    if len(response_text.split()) < 20: suggestions.append({"area": "Detail & Depth", "feedback": "Expand your answer with more detail or examples."})

    feedback_paragraphs = generate_openai_feedback(question_text, response_text)
    for p in feedback_paragraphs:
        suggestions.append({"area": "Feedback", "feedback": p})

    logger.info("Evaluation pipeline completed")
    return {
        "score": score10,
        "clarity_tier": get_tier(clr),
        "keyword_matches": matched,
        "expected_keywords": kws,
        "skills_shown": extract_skills_from_response(response_text),
        "relevance_breakdown": {"semantic": sem, "clarity": clr, "qterms": qterms},
        "improvement_suggestions": suggestions,
        "transcribed_text": response_text
    }
