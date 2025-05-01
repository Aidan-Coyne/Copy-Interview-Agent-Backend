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
from question_generation import extract_sectors  # kept for potential future use

# ─── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ─── Load models ────────────────────────────────────────────────────────────
embedder = SentenceTransformer('all-MiniLM-L6-v2')
nli      = pipeline("text-classification", model="roberta-large-mnli")
nlp      = spacy.load("en_core_web_sm")

def deep_round(value: float, ndigits: int) -> float:
    return round(value, ndigits)

class TranscriptionError(Exception):
    pass

# --------------------------------------------------
# Core utility functions
# --------------------------------------------------

def download_user_response_from_firebase(session_id: str, question_index: int, bucket: storage.Bucket) -> bytes:
    blob_path = f"sessions/{session_id}/audio_responses/response_{question_index+1}.mp3"
    blob = bucket.blob(blob_path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="Audio response not found in storage.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        blob.download_to_filename(tmp.name)
    data = open(tmp.name, "rb").read()
    try: os.remove(tmp.name)
    except: pass
    return data

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

def extract_company_keywords(company_info: str) -> List[str]:
    try:
        items = json.loads(company_info)
        kws = []
        for item in items:
            snippet = item.get("snippet", "").lower()
            doc = nlp(snippet)
            kws += [t.lemma_ for t in doc if t.pos_ in {"NOUN","VERB"}]
            kws += [e.text.lower() for e in doc.ents]
        return list(set(kws))
    except Exception as e:
        logger.error(f"Error extracting company keywords: {e}")
        return []

def extract_skills_from_response(response_text: str) -> List[str]:
    doc = nlp(response_text)
    return list({t.text.lower() for t in doc if t.pos_ in {"NOUN","VERB"}})

# --------------------------------------------------
# Keyword & Question-Type Extraction
# --------------------------------------------------

def get_relevant_keywords(
    question_data: Dict[str, Any],
    job_role: str,
    company_name: str,
    company_info: str
) -> Tuple[List[str], str, Optional[str]]:
    qtext  = question_data.get("question_text","").lower()
    skills = question_data.get("skills", [])
    exp    = question_data.get("experience", "")

    kws = set(extract_company_keywords(company_info))
    for s in skills:
        kws.add(nlp(s)[0].lemma_.lower())
    if exp: kws.add(nlp(exp)[0].lemma_.lower())
    if job_role: kws.add(nlp(job_role)[0].lemma_.lower())
    if company_name: kws.add(nlp(company_name)[0].lemma_.lower())
    relevant = [k for k in kws if len(k)>2]

    if "tell me about a time" in qtext or "describe a situation" in qtext:
        qtype = "behavioral"
    elif "how would you" in qtext:
        qtype = "situational"
    elif "skills" in qtext or "explain" in qtext:
        qtype = "technical"
    else:
        qtype = "general"

    try:
        sector = extract_sectors(json.loads(company_info))
    except:
        sector = None

    return relevant, qtype, sector

# --------------------------------------------------
# Relevance sub-scores
# --------------------------------------------------

def semantic_similarity(q: str, r: str) -> float:
    score = ((util.cos_sim(
        embedder.encode(q,True),
        embedder.encode(r,True)
    ).item() + 1) / 2) * 100
    logger.debug(f"Semantic similarity: {score:.1f}%")
    return score

def entailment_score(q: str, r: str) -> float:
    out = nli(f"{q} </s></s> {r}")
    ent = next((x for x in out if x["label"]=="ENTAILMENT"), None)
    score = ent["score"]*100 if ent else 0
    logger.debug(f"Entailment score: {score:.1f}%")
    return score

def tfidf_bm25_score(q: str, r: str) -> float:
    bm25 = BM25Okapi([r.lower().split()])
    sc   = bm25.get_scores(q.lower().split())[0]
    score= min(max(sc,0),1)*100
    logger.debug(f"BM25 score: {score:.1f}%")
    return score

def slot_match_score(q: str, r: str) -> float:
    qd, rd = nlp(q), nlp(r)
    slots = [t for t in qd if t.dep_ in ("ROOT","dobj")]
    if not slots: return 0.0
    matches = sum(1 for s in slots for t in rd if t.lemma_==s.lemma_)
    score = matches/len(slots)*100
    logger.debug(f"Slot match: {score:.1f}%")
    return score

# --------------------------------------------------
# Friendly tiered feedback templates
# --------------------------------------------------

TIERS = [(40, "needs_improvement"), (70, "on_track")]
def get_tier(score: float) -> str:
    for thresh, name in TIERS:
        if score < thresh:
            return name
    return "strong"

TEMPLATES = {
    "Topic Fit": {
        "needs_improvement": [
            "Let’s zero in on the question’s main topic—start by restating it to show alignment.",
            "Make sure your overall idea mirrors what the prompt is asking for."
        ],
        "on_track": [
            "Good overall match—tightening your phrasing will make it pop even more.",
            "You’ve got the topic right; lead with it to make it crystal clear."
        ],
        "strong": [
            "Great topic alignment! You could add an example to drive it home.",
            "Strong fit—try phrasing it as a one-sentence summary up front for extra polish."
        ]
    },
    "Clear Answer": {
        "needs_improvement": [
            "Start with a one-sentence summary to make your answer crystal clear.",
            "Kick off with ‘I would…’ so the reviewer immediately knows your approach."
        ],
        "on_track": [
            "Nice structure—consider moving your summary to the very first line.",
            "Good clarity—opening with a TL;DR will sharpen your message."
        ],
        "strong": [
            "Your answer is clear! You could add a quick example to illustrate it.",
            "Excellent clarity—lead with a bold statement of your plan for extra impact."
        ]
    },
    "Question Terms Used": {
        "needs_improvement": [
            "Weave in more of the exact words from the question to highlight relevance.",
            "Try sprinkling in the prompt’s keywords naturally—they’ll boost your fit."
        ],
        "on_track": [
            "Good use of key terms—see if you can sprinkle in a couple more for extra impact.",
            "You’ve hit most keywords; including one or two more will strengthen it."
        ],
        "strong": [
            "Great keyword coverage! Try swapping in a synonym to show range.",
            "Solid matching—alternating synonyms can show your vocabulary."
        ]
    },
    "Question Action & Object Answered": {
        "needs_improvement": [
            "Name the verb (action) and the thing you’re acting on in your first sentence.",
            "Cover both parts: what you’d do, and who/what you’d do it to."
        ],
        "on_track": [
            "You mentioned both parts—bringing them up front will elevate it.",
            "Nice coverage—leading with ‘I would [action] the [object]…’ is even stronger."
        ],
        "strong": [
            "Excellent coverage of action & object—try varying phrasing next time.",
            "Solid slot match—consider swapping order for variety."
        ]
    }
}

def pick_feedback(area: str, score: float) -> str:
    tier = get_tier(score)
    return random.choice(TEMPLATES.get(area, {}).get(tier, [""]))

# --------------------------------------------------
# Main scoring function
# --------------------------------------------------

def score_response(
    response_text: str,
    question_text: str,
    relevant_keywords: List[str],
    question_type: str,
    company_sector: Optional[str] = None
) -> Dict[str, Any]:
    # Keyword fuzzy match
    tokens = [t.lemma_.lower().strip(string.punctuation)
              for t in nlp(response_text) if not t.is_stop and len(t.text)>2]
    kws = [kw.lower().strip(string.punctuation) for kw in relevant_keywords]
    matched, missing = [], []
    for kw in kws:
        pct = max(fuzz.partial_ratio(kw, tok) for tok in tokens) if tokens else 0
        (matched if pct>=80 else missing).append(kw)
    keyword_score = len(matched)/len(kws)*100 if kws else 0

    # Sub-scores
    sem   = semantic_similarity(question_text, response_text)
    ent   = entailment_score(question_text, response_text)
    tfidf = tfidf_bm25_score(question_text, response_text)
    slot  = slot_match_score(question_text, response_text)

    # Combined relevance
    rel_w = {"semantic":.4,"entailment":.3,"tfidf":.15,"slot":.15}
    question_score = (sem*rel_w["semantic"] + ent*rel_w["entailment"] +
                      tfidf*rel_w["tfidf"] + slot*rel_w["slot"])

    # Final hybrid
    base_w = {"technical":(.5,.5),"behavioral":(.3,.7),"situational":(.3,.7)}
    kw_w, qt_w = base_w.get(question_type,(0.4,0.6))
    final_num = deep_round(kw_w*keyword_score + qt_w*question_score,2)
    score10 = round(final_num/10,1)

    # Build suggestions
    suggestions: List[Dict[str,str]] = []

    suggestions.append({
        "area": "Scoring Explanation",
        "feedback": f"Final score {score10}/10 (Keywords: {keyword_score:.1f}% × {kw_w}, Relevance: {question_score:.1f}% × {qt_w})."
    })
    suggestions.append({
        "area": "Relevance Breakdown",
        "feedback": (
            f"Topic Fit: {sem:.1f}%  •  "
            f"Clear Answer: {ent:.1f}%  •  "
            f"Question Terms Used: {tfidf:.1f}%  •  "
            f"Question Action & Object Answered: {slot:.1f}%"
        )
    })

    if sem < 70:
        suggestions.append({"area": "Topic Fit", "feedback": pick_feedback("Topic Fit", sem)})
    if ent < 70:
        suggestions.append({"area": "Clear Answer", "feedback": pick_feedback("Clear Answer", ent)})
    if tfidf < 70:
        suggestions.append({"area": "Question Terms Used", "feedback": pick_feedback("Question Terms Used", tfidf)})
    if slot < 70:
        suggestions.append({"area": "Question Action & Object Answered", "feedback": pick_feedback("Question Action & Object Answered", slot)})

    if missing:
        suggestions.append({
            "area": "Keyword Usage",
            "feedback": f"Consider adding missing keywords: {', '.join(missing)}."
        })

    if question_type == "behavioral":
        suggestions.append({"area":"Behavioral Structure","feedback":"Use STAR (Situation, Task, Action, Result)."})
    elif question_type == "situational":
        suggestions.append({"area":"Situational Strategy","feedback":"Outline your reasoning steps clearly."})
    elif question_type == "technical":
        suggestions.append({"area":"Technical Depth","feedback":"Include specific tools or examples."})

    if len(response_text.split()) < 20:
        suggestions.append({"area": "Detail & Depth", "feedback": "Expand with examples or explanations."})

    return {
        "score": score10,
        "keyword_matches": matched,
        "expected_keywords": kws,
        "skills_shown": extract_skills_from_response(response_text),
        "relevance_breakdown": {"semantic":sem,"entailment":ent,"tfidf":tfidf,"slot":slot},
        "improvement_suggestions": suggestions,
        "transcribed_text": response_text
    }

# --------------------------------------------------
# Debug runner
# --------------------------------------------------
if __name__ == "__main__":
    test_q = "Tell me about a time you led a team successfully."
    test_r = "I led five engineers, improved throughput by 20%, and delivered on time."
    kws, qtype, sec = get_relevant_keywords(
        {"question_text": test_q, "skills": ["leadership"], "experience": ""},
        "PM", "Acme", "[]"
    )
    import pprint
    pprint.pprint(score_response(test_r, test_q, kws, qtype, sec))
