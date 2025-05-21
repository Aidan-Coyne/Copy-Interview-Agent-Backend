import os
import json
import wave
import string
import logging
import subprocess
import random
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

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

def pick_feedback(area: str, score: float) -> str:
    tier = get_tier(score)
    return random.choice(TEMPLATES.get(area, {}).get(tier, [""]))

TIERS = [(40, "needs_improvement"), (70, "on_track")]
def get_tier(score: float) -> str:
    for thresh, name in TIERS:
        if score < thresh:
            return name
    return "strong"

TEMPLATES = {
    "Topic Fit": {
        "needs_improvement": ["Make sure your answer matches the main topic more closely."],
        "on_track": ["Good topic match—just tighten up the phrasing."],
        "strong": ["Excellent topic alignment!"]
    },
    "Clear Answer": {
        "needs_improvement": ["Try to reduce long pauses and keep focus."],
        "on_track": ["Your answer is clear—nice flow."],
        "strong": ["Excellent clarity and flow!"]
    },
    "Question Terms Used": {
        "needs_improvement": ["Include more of the question’s exact words to show relevance."],
        "on_track": ["Good use of key terms—add synonyms to show range."],
        "strong": ["Great keyword coverage!"]
    }
}

def generate_phi2_feedback(question: str, answer: str) -> List[str]:
    prompt = f"""
You are an interview coach helping a candidate improve their answer.

QUESTION:
"{question.strip()[:200]}"

ANSWER:
"{answer.strip()[:300]}"

Give two short paragraphs of feedback:
1. Mention one strong or effective part of the answer. Quote the exact phrase and say why it’s good.
2. Suggest one improvement. Be specific and avoid repeating the question.

Only return feedback. Do not repeat this prompt.
""".strip()

    logger.debug(f"🧠 Prompt size: {len(prompt)} chars")
    logger.debug(f"🧠 Prompt preview: {prompt[:200].replace(chr(10), ' ')}...")

    command = [
        "/llama/bin/llama",
        "-m", "/llama/models/phi-2.gguf",
        "-p", prompt,
        "-n", "100",
        "--top_k", "40",
        "--temp", "0.7"
    ]

    try:
        logger.debug(f"🚀 Executing: {' '.join(command)}")
        start = time.time()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "LD_LIBRARY_PATH": "/llama/bin"}
        )
        elapsed = time.time() - start
        logger.debug(f"✅ Subprocess completed in {elapsed:.2f}s (code {result.returncode})")

        if result.returncode != 0:
            logger.error("⚠️ LLM subprocess failed")
            logger.error(f"STDOUT:\n{result.stdout.strip()}")
            logger.error(f"STDERR:\n{result.stderr.strip()}")
            return ["Feedback could not be generated (model execution failed)."]

        lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        content = [line for line in lines if not line.lower().startswith("prompt:")]

        logger.debug(f"📤 Output preview: {content[:2]}")
        return content[:2] if content else ["No meaningful feedback was generated."]
    except subprocess.TimeoutExpired:
        logger.error("❌ Phi-2 subprocess timed out.")
        return ["Feedback generation took too long. Try a shorter answer."]
    except Exception:
        logger.exception("❌ Unexpected error during Phi-2 feedback generation")
        return ["Could not generate feedback. Please try again later."]

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

    if sem < 70: suggestions.append({"area": "Topic Fit", "feedback": pick_feedback("Topic Fit", sem)})
    if clr < 70: suggestions.append({"area": "Clear Answer", "feedback": pick_feedback("Clear Answer", clr)})
    if qterms < 70: suggestions.append({"area": "Question Terms Used", "feedback": pick_feedback("Question Terms Used", qterms)})
    if count_filler_words(response_text) > 0:
        suggestions.append({"area": "Clear Answer", "feedback": "Try to reduce filler words like 'uh', 'like', or 'you know'."})
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
        paragraphs = generate_phi2_feedback(question_text, response_text)
        for p in paragraphs:
            suggestions.append({"area": "Personalized Feedback", "feedback": p})
    except Exception:
        suggestions.append({"area": "Personalized Feedback", "feedback": "Could not generate feedback. Please try again later."})

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
