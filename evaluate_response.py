import os
import json
import wave
import string
import logging
import subprocess
import random
import time                              # ← for timing
from io import BytesIO
from typing import List, Dict, Tuple, Any, Optional

# ─── Vosk for offline ASR ─────────────────────────────────────────────────────
from vosk import Model, KaldiRecognizer

# ─── spaCy, Transformers & ONNX for scoring ─────────────────────────────────
import spacy
from fuzzywuzzy import fuzz
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, pipeline
import onnxruntime as ort
import numpy as np

# ─── VAD for pause detection ─────────────────────────────────────────────────
import webrtcvad

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ─── Load Vosk model once at import ──────────────────────────────────────────
VOSK_MODEL_PATH = "/app/models/vosk-model-small-en-us-0.15"
vosk_model = Model(VOSK_MODEL_PATH)
logger.info(f"✅ Loaded Vosk model from {VOSK_MODEL_PATH}")

# ─── ONNX semantic model ─────────────────────────────────────────────────────
MODEL_PATH = "/app/models/paraphrase-MiniLM-L3-v2.onnx"
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")
session = ort.InferenceSession(MODEL_PATH)

# ─── NLI & spaCy ─────────────────────────────────────────────────────────────
nli = pipeline("text-classification", model="roberta-large-mnli")
nlp = spacy.load("en_core_web_sm")

# ─── Dynamic feedback LLM ────────────────────────────────────────────────────
dynamic_feedback = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    tokenizer="google/flan-t5-small",
    max_length=300,
    do_sample=False
)

def deep_round(value: float, ndigits: int) -> float:
    return round(value, ndigits)

class TranscriptionError(Exception):
    pass

# ─── In-memory FFmpeg convert ─────────────────────────────────────────────────
def convert_to_wav(audio_data: bytes) -> bytes:
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"
        ],
        input=audio_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True
    )
    return proc.stdout

# ─── Vosk-based transcription ─────────────────────────────────────────────────
def transcribe_audio(audio_data: bytes) -> str:
    start = time.time()
    rec = KaldiRecognizer(vosk_model, 16000)
    rec.SetWords(True)
    if not rec.AcceptWaveform(audio_data):
        pass
    result_json = rec.Result()
    text = json.loads(result_json).get("text", "")
    elapsed = time.time() - start
    logger.info(f"⏱ Vosk transcription took {elapsed:.2f}s")
    if not text:
        raise TranscriptionError("Could not understand audio. Please speak clearly.")
    return text.lower()

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

# ─── ONNX embed + mean pooling ────────────────────────────────────────────────
def encode(text: str) -> np.ndarray:
    tokens = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    ort_inputs = {k: tokens[k] for k in ("input_ids", "attention_mask")}
    outputs = session.run(None, ort_inputs)
    last_hidden = outputs[0]           # (1,128,384)
    mask = tokens["attention_mask"][..., None]
    masked_hidden = last_hidden * mask
    summed = masked_hidden.sum(axis=1)
    lengths = mask.sum(axis=1)
    mean_pooled = summed / lengths
    return mean_pooled[0]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def semantic_similarity(q: str, r: str) -> float:
    vec_q = encode(q)
    vec_r = encode(r)
    score = ((cosine_similarity(vec_q, vec_r) + 1) / 2) * 100
    logger.debug(f"Semantic similarity: {score:.1f}%")
    return score

# ─── NEW: long-pause detection using WebRTC VAD ──────────────────────────────
def detect_long_pauses(wav_bytes: bytes) -> float:
    """Return approx proportion of silence in the response (0.0–1.0)."""
    vad = webrtcvad.Vad(2)
    try:
        wf = wave.open(BytesIO(wav_bytes), 'rb')
    except Exception as e:
        logger.warning(f"Pause detection skipped (could not read WAV): {e}")
        return 0.0

    if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
        logger.warning("Pause detection requires 16 kHz mono PCM; skipping")
        return 0.0

    frame_duration_ms = 30
    frames: List[bytes] = []
    num_speech = 0

    while True:
        data = wf.readframes(int(wf.getframerate() * frame_duration_ms / 1000.0))
        if not data or len(data) < wf.getsampwidth():
            break
        is_speech = vad.is_speech(data, wf.getframerate())
        num_speech += int(is_speech)
        frames.append(data)

    total = len(frames)
    if total == 0:
        return 0.0
    silence_ratio = 1 - (num_speech / total)
    logger.debug(f"Detected silence ratio: {silence_ratio:.2f}")
    return silence_ratio

def clarity_score(q: str, r: str, wav_bytes: bytes) -> float:
    """
    Combines NLI entailment with a long-pause penalty.
    """
    out = nli(f"{q} </s></s> {r}")
    ent = next((x for x in out if x["label"] == "ENTAILMENT"), None)
    entail_pct = (ent["score"] * 100) if ent else 0.0

    # penalise long pauses
    pause_ratio = detect_long_pauses(wav_bytes)
    clarity = entail_pct * (1 - pause_ratio)
    logger.debug(f"Clarity before pause penalty: {entail_pct:.1f}%, pause_ratio: {pause_ratio:.2f}, final: {clarity:.1f}%")
    return clarity

def tfidf_bm25_score(q: str, r: str) -> float:
    bm25 = BM25Okapi([r.lower().split()])
    sc = bm25.get_scores(q.lower().split())[0]
    score = min(max(sc, 0), 1) * 100
    logger.debug(f"BM25 score: {score:.1f}%")
    return score

# ─── Templates ────────────────────────────────────────────────────────────────
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

# ─── Main scoring + dynamic LLM feedback ─────────────────────────────────────
def score_response(
    response_text: str,
    question_text: str,
    relevant_keywords: List[str],
    question_type: str,
    company_sector: Optional[str] = None
) -> Dict[str, Any]:
    start_time = time.time()
    logger.info("⏳ Starting evaluation pipeline")

    # Keyword matching
    tokens = [
        t.lemma_.lower().strip(string.punctuation)
        for t in nlp(response_text)
        if not t.is_stop and len(t.text) > 2
    ]
    kws = [kw.lower().strip(string.punctuation) for kw in relevant_keywords]
    matched, missing = [], []
    for kw in kws:
        pct = max(fuzz.partial_ratio(kw, tok) for tok in tokens) if tokens else 0
        (matched if pct >= 80 else missing).append(kw)
    keyword_score = len(matched) / len(kws) * 100 if kws else 0

    # Subscores
    sem   = semantic_similarity(question_text, response_text)
    # pass the raw WAV bytes through so clarity_score can detect pauses
    # assume you passed wav_bytes into this function or stored globally
    wav_bytes = globals().get("latest_wav_bytes", b"")
    clr   = clarity_score      (question_text, response_text, wav_bytes)
    tfidf = tfidf_bm25_score   (question_text, response_text)

    # Composite (semantic, clarity, tfidf)
    rel_w = {"semantic": .8, "clarity": .10, "tfidf": .10}
    question_score = (
        sem * rel_w["semantic"] +
        clr * rel_w["clarity"]   +
        tfidf * rel_w["tfidf"]
    )

    # Final 0–10
    base_w = {"technical": (.4, .6), "behavioral": (.2, .8), "situational": (.2, .8)}
    kw_w, qt_w = base_w.get(question_type, (0.2, 0.8))
    final_num = deep_round(kw_w * keyword_score + qt_w * question_score, 2)
    score10   = round(final_num / 10, 1)

    # Static suggestions
    suggestions: List[Dict[str, str]] = [
        {
            "area": "Scoring Explanation",
            "feedback": f"Final score {score10}/10 (Keywords {keyword_score:.1f}%×{kw_w}, Relevance {question_score:.1f}%×{qt_w})."
        },
        {
            "area": "Relevance Breakdown",
            "feedback": (
                f"Topic Fit: {sem:.1f}%  •  Clear Answer: {clr:.1f}%  •  "
                f"Question Terms Used: {tfidf:.1f}%"
            )
        }
    ]
    if sem   < 70: suggestions.append({"area":"Topic Fit",           "feedback": pick_feedback("Topic Fit", sem)})
    if clr   < 70: suggestions.append({"area":"Clear Answer",        "feedback": pick_feedback("Clear Answer", clr)})
    if tfidf < 70: suggestions.append({"area":"Question Terms Used", "feedback": pick_feedback("Question Terms Used", tfidf)})
    if missing:
        suggestions.append({"area":"Keyword Usage", "feedback": f"Consider adding missing keywords: {', '.join(missing)}."})
    if question_type == "behavioral":
        suggestions.append({"area":"Behavioral Structure","feedback":"Use STAR (Situation, Task, Action, Result)."})
    elif question_type == "situational":
        suggestions.append({"area":"Situational Strategy","feedback":"Outline your reasoning steps clearly."})
    elif question_type == "technical":
        suggestions.append({"area":"Technical Depth","feedback":"Include specific tools or concrete examples."})
    if len(response_text.split()) < 20:
        suggestions.append({"area":"Detail & Depth","feedback":"Expand with examples or explanations."})

    # ─── Dynamic, few-shot LLM feedback ─────────────────────────────────────────
    try:
        prompt = (
            "You are an expert interview coach. Be specific and actionable.\n"
            f"Question: “{question_text}”\n"
            f"Answer: “{response_text}”\n"
            f"Metrics: semantic={sem:.1f}%, clarity={clr:.1f}%, tfidf={tfidf:.1f}%\n\n"
            "Produce:\n"
            "• 2–3 sentence summary of strengths (cite the candidate’s wording).\n"
            "• 2–3 sentence recommendations for improvement (concrete next steps).\n"
            "• 1 example of an improved phrasing.\n"
        )
        llm_out = dynamic_feedback(prompt)[0]["generated_text"].strip()
        logger.debug(f"LLM returned:\n{llm_out}")
        suggestions.append({
            "area": "Personalized Feedback",
            "feedback": llm_out
        })
    except Exception:
        logger.exception("Failed to generate dynamic feedback")

    # ─── Final timing & suggestions dump ────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info(f"✅ Evaluation pipeline completed in {elapsed:.2f}s")
    logger.debug(f"Full suggestions list: {suggestions}")

    return {
        "score": score10,
        "clarity_tier": get_tier(clr),
        "keyword_matches": matched,
        "expected_keywords": kws,
        "skills_shown": extract_skills_from_response(response_text),
        "relevance_breakdown": {
            "semantic": sem,
            "clarity":  clr,
            "tfidf":    tfidf
        },
        "improvement_suggestions": suggestions,
        "transcribed_text": response_text
    }
