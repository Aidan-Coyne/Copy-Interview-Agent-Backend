import os
import uuid
import json
import wave
import string
import logging
import subprocess
import tempfile
import random
from io import BytesIO
from typing import List, Dict, Tuple, Any, Optional

import speech_recognition as sr
import spacy
from fuzzywuzzy import fuzz
from fastapi import HTTPException
from google.cloud import storage
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

embedder = SentenceTransformer('all-MiniLM-L6-v2')
nli      = pipeline("text-classification", model="roberta-large-mnli")
nlp      = spacy.load("en_core_web_sm")

def deep_round(value: float, ndigits: int) -> float:
    return round(value, ndigits)

class TranscriptionError(Exception):
    pass

def convert_to_wav(audio_data: bytes) -> bytes:
    in_fn  = f"temp_{uuid.uuid4().hex}.webm"
    out_fn = f"temp_{uuid.uuid4().hex}.wav"
    with open(in_fn, "wb") as f: f.write(audio_data)
    subprocess.run(
        ["ffmpeg", "-y", "-i", in_fn, "-acodec", "pcm_s16le", out_fn],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    wav = open(out_fn, "rb").read()
    for fn in (in_fn, out_fn):
        try: os.remove(fn)
        except: pass
    return wav

def transcribe_audio(audio_data: bytes) -> str:
    wf = wave.open(BytesIO(audio_data), 'rb')
    audio = sr.AudioData(audio_data, wf.getframerate(), wf.getsampwidth())
    recognizer = sr.Recognizer()
    try:
        return recognizer.recognize_google(audio).lower()
    except sr.UnknownValueError:
        raise TranscriptionError("Could not understand audio. Please speak clearly.")
    except sr.RequestError as e:
        raise TranscriptionError(f"Speech recognition error: {e}")

def extract_skills_from_response(response_text: str) -> List[str]:
    doc = nlp(response_text)
    return list({t.text.lower() for t in doc if t.pos_ in {"NOUN","VERB"}})

# ─── Updated Function: Only pass role & sector keywords ───────────────────────

def get_relevant_keywords(
    question_data: Dict[str, Any],
    job_role: str,
    company_name: str,
    company_info: str
) -> Tuple[List[str], str, Optional[str]]:
    """
    Returns:
        - Relevant keywords (ONLY role_keywords + sector_keywords).
        - Detected question type.
        - Sector (unused now, returns None).
    """
    qtext = question_data.get("question_text", "").lower()

    # Only include role and sector keywords
    kws = set()
    for rk in question_data.get("role_keywords", []):
        kws.add(nlp(rk)[0].lemma_.lower())
    for sk in question_data.get("sector_keywords", []):
        kws.add(nlp(sk)[0].lemma_.lower())

    relevant = [k for k in kws if len(k) > 2]

    # Heuristic type detection
    if "tell me about a time" in qtext or "describe a situation" in qtext:
        qtype = "behavioral"
    elif "how would you" in qtext:
        qtype = "situational"
    elif "skills" in qtext or "explain" in qtext:
        qtype = "technical"
    else:
        qtype = "general"

    return relevant, qtype, None

# ─── Scoring and Feedback ─────────────────────────────────────────────────────

def semantic_similarity(q: str, r: str) -> float:
    score = ((util.cos_sim(embedder.encode(q, True), embedder.encode(r, True)).item() + 1) / 2) * 100
    logger.debug(f"Semantic similarity: {score:.1f}%")
    return score

def clarity_score(q: str, r: str) -> float:
    out = nli(f"{q} </s></s> {r}")
    ent = next((x for x in out if x["label"] == "ENTAILMENT"), None)
    entail_pct = (ent["score"] * 100) if ent else 0.0

    first_line = r.strip().split("\n", 1)[0].lower()
    if first_line.startswith(("i would", "i'll", "i will", "my approach", "my plan")):
        return max(entail_pct, 60.0)
    return entail_pct

def tfidf_bm25_score(q: str, r: str) -> float:
    bm25 = BM25Okapi([r.lower().split()])
    sc   = bm25.get_scores(q.lower().split())[0]
    score = min(max(sc, 0), 1) * 100
    logger.debug(f"BM25 score: {score:.1f}%")
    return score

def slot_match_score(q: str, r: str) -> float:
    qd, rd = nlp(q), nlp(r)
    slots = [t for t in qd if t.dep_ in ("ROOT", "dobj")]
    if not slots: return 0.0
    matches = sum(1 for s in slots for t in rd if t.lemma_ == s.lemma_)
    score = matches / len(slots) * 100
    logger.debug(f"Slot match: {score:.1f}%")
    return score

# ─── Tier & Template-Based Feedback ───────────────────────────────────────────

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
        "needs_improvement": ["Start with a direct one-sentence summary of your plan."],
        "on_track": ["Your answer is clear—consider opening with “I would…” for extra punch."],
        "strong": ["Very clear answer! You could add an illustrative example next time."]
    },
    "Question Terms Used": {
        "needs_improvement": ["Include more of the question’s exact words to show relevance."],
        "on_track": ["Nice use of key terms—add a synonym or two to show range."],
        "strong": ["Great keyword coverage!"]
    },
    "Question Action & Object Answered": {
        "needs_improvement": ["Mention both the action and the object right away (e.g. “I would do X to Y…”)."],
        "on_track": ["Good action & object usage—bring them up front."],
        "strong": ["Excellent coverage of action & object!"]
    }
}

def pick_feedback(area: str, score: float) -> str:
    tier = get_tier(score)
    return random.choice(TEMPLATES.get(area, {}).get(tier, [""]))

# ─── Main Evaluation Pipeline ─────────────────────────────────────────────────

def score_response(
    response_text: str,
    question_text: str,
    relevant_keywords: List[str],
    question_type: str,
    company_sector: Optional[str] = None
) -> Dict[str, Any]:
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

    sem   = semantic_similarity(question_text, response_text)
    clr   = clarity_score      (question_text, response_text)
    tfidf = tfidf_bm25_score   (question_text, response_text)
    slot  = slot_match_score   (question_text, response_text)

    rel_w = {"semantic": .7, "clarity": .15, "tfidf": .10, "slot": .5}
    question_score = (
        sem * rel_w["semantic"]
        + clr * rel_w["clarity"]
        + tfidf * rel_w["tfidf"]
        + slot * rel_w["slot"]
    )

    base_w = {"technical": (.4, .6), "behavioral": (.2, .8), "situational": (.2, .8)}
    kw_w, qt_w = base_w.get(question_type, (0.2, 0.8))
    final_num = deep_round(kw_w * keyword_score + qt_w * question_score, 2)
    score10   = round(final_num / 10, 1)

    suggestions: List[Dict[str, str]] = []
    suggestions.append({
        "area": "Scoring Explanation",
        "feedback": f"Final score {score10}/10 (Keywords {keyword_score:.1f}%×{kw_w}, Relevance {question_score:.1f}%×{qt_w})."
    })
    suggestions.append({
        "area": "Relevance Breakdown",
        "feedback": (
            f"Topic Fit: {sem:.1f}%  •  Clear Answer: {clr:.1f}%  •  "
            f"Question Terms Used: {tfidf:.1f}%  •  Question Action & Object Answered: {slot:.1f}%"
        )
    })

    if sem   < 70: suggestions.append({"area": "Topic Fit", "feedback": pick_feedback("Topic Fit", sem)})
    if clr   < 70: suggestions.append({"area": "Clear Answer", "feedback": pick_feedback("Clear Answer", clr)})
    if tfidf < 70: suggestions.append({"area": "Question Terms Used", "feedback": pick_feedback("Question Terms Used", tfidf)})
    if slot  < 70: suggestions.append({"area": "Question Action & Object Answered", "feedback": pick_feedback("Question Action & Object Answered", slot)})

    if missing:
        suggestions.append({
            "area": "Keyword Usage",
            "feedback": f"Consider adding missing keywords: {', '.join(missing)}."
        })

    if question_type == "behavioral":
        suggestions.append({"area": "Behavioral Structure", "feedback": "Use STAR (Situation, Task, Action, Result)."})
    elif question_type == "situational":
        suggestions.append({"area": "Situational Strategy", "feedback": "Outline your reasoning steps clearly."})
    elif question_type == "technical":
        suggestions.append({"area": "Technical Depth", "feedback": "Include specific tools or concrete examples."})

    if len(response_text.split()) < 20:
        suggestions.append({"area": "Detail & Depth", "feedback": "Expand with examples or explanations."})

    return {
        "score": score10,
        "clarity_tier": get_tier(clr),
        "keyword_matches": matched,
        "expected_keywords": kws,
        "skills_shown": extract_skills_from_response(response_text),
        "relevance_breakdown": {"semantic": sem, "clarity": clr, "tfidf": tfidf, "slot": slot},
        "improvement_suggestions": suggestions,
        "transcribed_text": response_text
    }
