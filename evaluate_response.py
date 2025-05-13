import os
import json
import wave
import string
import logging
import subprocess
import random
import time
import re
from io import BytesIO
from typing import List, Dict, Tuple, Any, Optional

from vosk import Model, KaldiRecognizer
import spacy
from fuzzywuzzy import fuzz
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, pipeline
import onnxruntime as ort
import numpy as np
import webrtcvad

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

VOSK_MODEL_PATH = "/app/models/vosk-model-en-us-0.22-lgraph"
vosk_model = Model(VOSK_MODEL_PATH)
logger.info(f"✅ Loaded Vosk model from {VOSK_MODEL_PATH}")

MODEL_PATH = "/app/models/paraphrase-MiniLM-L3-v2.onnx"
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")
session = ort.InferenceSession(MODEL_PATH)

nli = pipeline("text-classification", model="roberta-large-mnli")
nlp = spacy.load("en_core_web_sm")

dynamic_feedback = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_length=512,
    do_sample=False
)

FILLER_WORDS = {
    "um", "uh", "like", "you know", "i mean", "basically", "sort of", "kind of", "er", "erm"
}

def deep_round(value: float, ndigits: int) -> float:
    return round(value, ndigits)

class TranscriptionError(Exception):
    pass

def convert_to_wav(audio_data: bytes) -> bytes:
    # Merged: format conversion + denoising
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
        wf = wave.open(BytesIO(audio_data), 'rb')
    except Exception:
        raise TranscriptionError("Invalid audio format.")

    if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
        raise TranscriptionError("Audio must be mono, 16-bit, 16kHz WAV format.")

    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    results = []

    while True:
        buf = wf.readframes(4000)
        if len(buf) == 0:
            break
        if rec.AcceptWaveform(buf):
            part = json.loads(rec.Result())
            results.append(part.get("text", ""))
    results.append(json.loads(rec.FinalResult()).get("text", ""))
    text = " ".join(results).strip()

    elapsed = time.time() - start
    logger.info(f"⏱ Vosk transcription took {elapsed:.2f}s")
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

def get_relevant_keywords(
    question_data: Dict[str, Any],
    job_role: str,
    company_name: str,
    company_info: str
) -> Tuple[List[str], str, Optional[str]]:
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

def clarity_score(q: str, r: str, wav_bytes: bytes) -> float:
    out = nli(f"{q} </s></s> {r}")
    ent = next((x for x in out if x["label"] == "ENTAILMENT"), None)
    entail = (ent["score"] * 100) if ent else 0.0

    pause_penalty = detect_long_pauses(wav_bytes)
    filler_count = count_filler_words(r)
    filler_penalty = min(0.05 * filler_count, 0.25)  # Max 25% deduction

    clarity = entail * (1 - pause_penalty)
    adjusted = clarity * (1 - filler_penalty)
    return adjusted

TIERS = [(40, "needs_improvement"), (70, "on_track")]
def get_tier(score: float) -> str:
    for thresh, name in TIERS:
        if score < thresh:
            return name
    return "strong"

TEMPLATES = {
    "Topic Fit": {
        "needs_improvement": ["Make sure your answer matches the main topic more closely."],
        "on_track":          ["Good topic match—just tighten up the phrasing."],
        "strong":            ["Excellent topic alignment!"]
    },
    "Clear Answer": {
        "needs_improvement": ["Try to reduce long pauses and keep focus."],
        "on_track":          ["Your answer is clear—nice flow."],
        "strong":            ["Excellent clarity and flow!"]
    },
    "Question Terms Used": {
        "needs_improvement": ["Include more of the question’s exact words to show relevance."],
        "on_track":          ["Good use of key terms—add synonyms to show range."],
        "strong":            ["Great keyword coverage!"]
    }
}

def pick_feedback(area: str, score: float) -> str:
    tier = get_tier(score)
    return random.choice(TEMPLATES.get(area, {}).get(tier, [""]))

def score_response(
    response_text: str,
    question_text: str,
    relevant_keywords: List[str],
    question_type: str,
    company_sector: Optional[str] = None,
    wav_bytes: bytes = b""
) -> Dict[str, Any]:
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
    clr = clarity_score(question_text, response_text, wav_bytes)
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
    if sem < 70: suggestions.append({"area": "Topic Fit", "feedback": pick_feedback("Topic Fit", sem)})
    if clr < 70: suggestions.append({"area": "Clear Answer", "feedback": pick_feedback("Clear Answer", clr)})
    if qterms < 70: suggestions.append({"area": "Question Terms Used", "feedback": pick_feedback("Question Terms Used", qterms)})
    if count_filler_words(response_text) > 0:
        suggestions.append({"area": "Clear Answer", "feedback": "Try to reduce filler words like 'uh', 'like', or 'you know' to improve clarity."})
    if missing:
        suggestions.append({"area": "Keyword Usage", "feedback": f"Consider adding missing keywords: {', '.join(missing)}."})
    if question_type == "behavioral":
        suggestions.append({"area": "Behavioral Structure", "feedback": "Use STAR (Situation, Task, Action, Result)."})
    elif question_type == "situational":
        suggestions.append({"area": "Situational Strategy", "feedback": "Outline your reasoning steps clearly."})
    elif question_type == "technical":
        suggestions.append({"area": "Technical Depth", "feedback": "Include specific tools or concrete examples."})
    if len(response_text.split()) < 20:
        suggestions.append({"area": "Detail & Depth", "feedback": "Expand with examples or explanations."})

    try:
        prompt = f"""
You are an interview coach. A candidate just answered the question below.

QUESTION:
"{question_text}"

ANSWER:
"{response_text}"

Your task is to provide **2 helpful and specific bullet points**:
1. Highlight something strong or relevant in the answer. Quote the exact phrase from the answer if possible, and explain why it's good.
2. Suggest how they could improve this specific answer. Be direct and specific—suggest missing details, better examples, or structure.

Make sure the feedback is focused **only on this question and answer**. Be concise, helpful, and constructive.

Return exactly:
1. [positive feedback]
2. [improvement suggestion]
"""
        llm_out = dynamic_feedback(prompt.strip())[0]["generated_text"].strip()
        logger.debug(f"LLM returned raw output:\n{llm_out}")

        bullets = [line.strip("• ").strip() for line in llm_out.splitlines() if line.strip()]
        valid_bullets = [b for b in bullets if b and not b.strip().isdigit()]
        if not valid_bullets:
            suggestions.append({
                "area": "Personalized Feedback",
                "feedback": "Could not extract meaningful personalized feedback. Try speaking more clearly or giving a fuller answer."
            })
        else:
            for item in valid_bullets[:2]:
                suggestions.append({"area": "Personalized Feedback", "feedback": item})

    except Exception:
        logger.exception("Failed to generate dynamic feedback")

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
