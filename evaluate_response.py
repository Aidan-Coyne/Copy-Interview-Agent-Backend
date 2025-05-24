import os
import json
import wave
import string
import logging
import subprocess
import time
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

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ✅ Initialize OpenAI Client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Load FasterWhisper tiny model
whisper_model = WhisperModel("tiny", compute_type="int8")
logger.info("✅ Loaded FasterWhisper model: tiny")

# ✅ Load ONNX sentence encoder
MODEL_PATH = "/app/models/paraphrase-MiniLM-L3-v2.onnx"
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")
session = ort.InferenceSession(MODEL_PATH)

# ✅ Load SpaCy
nlp = spacy.load("en_core_web_sm")

FILLER_WORDS = {
    "um", "uh", "like", "you know", "i mean", "basically", "sort of", "kind of", "er", "erm"
}

def deep_round(value: float, ndigits: int) -> float:
    return round(value, ndigits)

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
    start = time.time()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            temp_wav.write(audio_data)
            temp_wav_path = temp_wav.name

        segments, _ = whisper_model.transcribe(temp_wav_path, beam_size=1)
        text = " ".join(segment.text for segment in segments).strip()
    except Exception as e:
        raise TranscriptionError(f"FasterWhisper transcription failed: {e}")

    elapsed = time.time() - start
    logger.info(f"⏱ FasterWhisper transcription took {elapsed:.2f}s")

    if not text:
        raise TranscriptionError("Could not understand audio. Please speak clearly.")
    return text.lower()

def count_filler_words(text: str) -> int:
    normalized = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    words = normalized.split()
    return sum(1 for word in words if word in FILLER_WORDS)

def extract_skills_from_response(response_text: str) -> List[str]:
    doc = nlp(response_text)
    return list({t.text.lower() for t in doc if t.pos_ in {"NOUN", "VERB"}})

def get_relevant_keywords(question_data: Dict[str, Any], job_role: str, company_name: str, company_info: str) -> Tuple[List[str], str, Optional[str]]:
    qtext = question_data.get("question_text", "").lower()
    kws = set()
    for rk in question_data.get("role_keywords", []):
        kws.add(nlp(rk)[0].lemma_.lower())
    for sk in question_data.get("sector_keywords", []):
        kws.add(nlp(sk)[0].lemma_.lower())
    relevant = [k for k in kws if len(k) > 2]

    if "tell me about a time" in qtext or "describe a situation" in qtext:
        qtype = "behavioral"
    elif "how would you" in qtext:
        qtype = "situational"
    elif "skills" in qtext or "explain" in qtext:
        qtype = "technical"
    else:
        qtype = "general"
    return relevant, qtype, None

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
    summed = (last_hidden * mask).sum(axis=1)
    lengths = mask.sum(axis=1)
    return (summed / lengths)[0]

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
    except Exception:
        return 0.0

    frame_ms = 30
    frames, speech = 0, 0
    while True:
        data = wf.readframes(int(wf.getframerate() * frame_ms / 1000))
        if not data or len(data) < wf.getsampwidth():
            break
        if vad.is_speech(data, wf.getframerate()):
            speech += 1
        frames += 1

    return 1 - (speech / frames) if frames else 0.0

def clarity_score(r: str, wav_bytes: bytes) -> float:
    pause_penalty = detect_long_pauses(wav_bytes)
    filler_count = count_filler_words(r)
    filler_penalty = min(0.05 * filler_count, 0.25)
    return 100 * (1 - pause_penalty) * (1 - filler_penalty)

def generate_openai_feedback(question: str, answer: str) -> List[str]:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI career coach providing structured interview feedback."},
                {"role": "user", "content": f"""You are evaluating a candidate's spoken interview answer.

Question:
\"{question}\"

Answer:
\"{answer}\"

Provide structured feedback:
1. What they did well (quote strong phrases).
2. What could be improved (clarify, expand, structure).
Keep it supportive, concise, and actionable."""}
            ],
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip().split("\n\n")
    except Exception as e:
        logger.error(f"OpenAI feedback failed: {e}")
        return ["(LLM Feedback error)"]

def get_tier(score: float) -> str:
    if score < 40:
        return "needs_improvement"
    elif score < 70:
        return "on_track"
    return "strong"

def score_response(response_text: str, question_text: str, relevant_keywords: List[str], question_type: str, company_sector: Optional[str] = None, wav_bytes: bytes = b"") -> Dict[str, Any]:
    start_time = time.time()
    logger.info("⏳ Starting evaluation pipeline")

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
    base_w = {"technical": (.4, .6), "behavioral": (.2, .8), "situational": (.2, .8)}
    kw_w, qt_w = base_w.get(question_type, (0.2, 0.8))
    final_num = deep_round(kw_w * keyword_score + qt_w * question_score, 2)
    score10 = round(final_num / 10, 1)

    suggestions = [
        {"area": "Scoring Explanation", "feedback": f"Final score {score10}/10 (Keywords {keyword_score:.1f}%×{kw_w}, Relevance {question_score:.1f}%×{qt_w})."},
        {"area": "Relevance Breakdown", "feedback": f"Topic Fit: {sem:.1f}%  •  Clarity: {clr:.1f}%  •  Question Terms Used: {qterms:.1f}%"}
    ]

    if sem < 70:
        suggestions.append({"area": "Topic Fit", "feedback": "Consider aligning your answer more closely with the main topic."})
    if clr < 70:
        suggestions.append({"area": "Clear Answer", "feedback": "Try to reduce long pauses and avoid filler words."})
    if qterms < 70:
        suggestions.append({"area": "Question Terms Used", "feedback": "Use key terms from the question to demonstrate focus."})
    if count_filler_words(response_text) > 0:
        suggestions.append({"area": "Clear Answer", "feedback": "Reduce filler words like 'uh', 'like', or 'you know'."})
    if missing:
        suggestions.append({"area": "Keyword Usage", "feedback": f"Consider adding missing keywords: {', '.join(missing)}."})
    if len(response_text.split()) < 20:
        suggestions.append({"area": "Detail & Depth", "feedback": "Expand your answer with more detail or examples."})

    paragraphs = generate_openai_feedback(question_text, response_text)
    for p in paragraphs:
        suggestions.append({"area": "Personalized Feedback", "feedback": p})

    logger.info(f"✅ Evaluation pipeline completed in {time.time() - start_time:.2f}s")
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
