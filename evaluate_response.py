import os
import uuid
import json
import wave
import string
import logging
import subprocess
import tempfile
from io import BytesIO
from typing import List, Dict, Tuple, Any, Optional

import speech_recognition as sr
import spacy
from fuzzywuzzy import fuzz
from fastapi import HTTPException
from google.cloud import storage

# Optional: install rank-bm25 via pip install rank-bm25
from rank_bm25 import BM25Okapi

# NEW imports for semantic relevance
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from question_generation import extract_sectors  # kept for potential future use

# Logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Helper rounding function
def deep_round(value: float, ndigits: int) -> float:
    """Round a float to a given precision."""
    return round(value, ndigits)

# Load models once at import time
embedder = SentenceTransformer('all-MiniLM-L6-v2')
nli = pipeline("text-classification", model="roberta-large-mnli")
nlp = spacy.load("en_core_web_sm")


class TranscriptionError(Exception):
    pass


# --------------------------------------------------
# Core utility functions
# --------------------------------------------------

def download_user_response_from_firebase(session_id: str, question_index: int, bucket: storage.Bucket) -> bytes:
    logger.debug(f"Downloading audio for session={session_id}, question_index={question_index}")
    blob_path = f"sessions/{session_id}/audio_responses/response_{question_index + 1}.mp3"
    blob = bucket.blob(blob_path)
    if not blob.exists():
        logger.error(f"Audio response not found: {blob_path}")
        raise HTTPException(status_code=404, detail="Audio response not found in storage.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        blob.download_to_filename(temp_file.name)
        logger.debug(f"Downloaded to temp file: {temp_file.name}")
    with open(temp_file.name, "rb") as f:
        audio_bytes = f.read()
    try:
        os.remove(temp_file.name)
    except Exception as e:
        logger.warning(f"Failed to delete temp file: {e}")
    return audio_bytes


def convert_to_wav(audio_data: bytes) -> bytes:
    input_filename = f"temp_{uuid.uuid4().hex}.webm"
    output_filename = f"temp_{uuid.uuid4().hex}.wav"
    try:
        with open(input_filename, "wb") as f:
            f.write(audio_data)
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_filename, "-acodec", "pcm_s16le", output_filename],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        with open(output_filename, "rb") as f:
            wav_data = f.read()
    finally:
        for fn in (input_filename, output_filename):
            if os.path.exists(fn):
                os.remove(fn)
    return wav_data


def transcribe_audio(audio_data: bytes) -> str:
    recognizer = sr.Recognizer()
    try:
        with wave.open(BytesIO(audio_data), 'rb') as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            logger.info(f"Audio properties: sample_rate={sample_rate}, sample_width={sample_width}")
        audio_data_sr = sr.AudioData(audio_data, sample_rate=sample_rate, sample_width=sample_width)
        text = recognizer.recognize_google(audio_data_sr)
        return text.lower()
    except sr.UnknownValueError:
        raise TranscriptionError("Could not understand audio. Please speak clearly.")
    except sr.RequestError as e:
        raise TranscriptionError(f"Speech recognition error: {e}")
    except Exception as e:
        raise TranscriptionError(f"Error transcribing audio: {e}")


def extract_company_keywords(company_info: str) -> List[str]:
    try:
        items = json.loads(company_info)
        kw = []
        for item in items:
            snippet = item.get("snippet", "").lower()
            doc = nlp(snippet)
            kw.extend(token.lemma_ for token in doc if token.pos_ in {"NOUN", "VERB"})
            kw.extend(ent.text.lower() for ent in doc.ents)
        return list(set(kw))
    except Exception as e:
        logger.error(f"Error extracting company keywords: {e}")
        return []


def extract_skills_from_response(response_text: str) -> List[str]:
    doc = nlp(response_text)
    return list(set(token.text.lower() for token in doc if token.pos_ in {"NOUN", "VERB"}))


# --------------------------------------------------
# Keyword & Question-Type Extraction
# --------------------------------------------------

def get_relevant_keywords(
    question_data: Dict[str, Any],
    job_role: str,
    company_name: str,
    company_info: str
) -> Tuple[List[str], str, Optional[str]]:
    """
    Returns:
      - relevant_keywords: list of lemmas
      - question_type: "behavioral", "situational", "technical", or "general"
      - company_sector: Optional[str]
    """
    qtext = question_data.get("question_text", "").lower()
    skills = question_data.get("skills", [])
    exp = question_data.get("experience", "")

    kws = set()
    for s in skills:
        kws.add(nlp(s)[0].lemma_.lower())
    if exp:
        kws.add(nlp(exp)[0].lemma_.lower())
    if job_role:
        kws.add(nlp(job_role)[0].lemma_.lower())
    if company_name:
        kws.add(nlp(company_name)[0].lemma_.lower())

    # add company-specific keywords
    kws.update(extract_company_keywords(company_info))

    relevant = [kw for kw in kws if len(kw) > 2]

    # determine question type
    if "tell me about a time" in qtext or "describe a situation" in qtext:
        qtype = "behavioral"
    elif "how would you" in qtext:
        qtype = "situational"
    elif "skills" in qtext or "explain" in qtext:
        qtype = "technical"
    else:
        qtype = "general"

    # extract sector if needed (no longer used)
    try:
        ci = json.loads(company_info)
        sector = extract_sectors(ci)
    except Exception:
        sector = None

    return relevant, qtype, sector


# --------------------------------------------------
# Relevance sub-scores
# --------------------------------------------------

def semantic_similarity(q: str, r: str) -> float:
    q_emb = embedder.encode(q, convert_to_tensor=True)
    r_emb = embedder.encode(r, convert_to_tensor=True)
    score = (util.cos_sim(q_emb, r_emb).item() + 1) / 2 * 100
    logger.debug(f"Semantic similarity: {score:.2f}")
    return score


def entailment_score(question: str, response: str) -> float:
    out = nli(f"{question} </s></s> {response}")
    ent = next((x for x in out if x['label'] == 'ENTAILMENT'), None)
    score = ent['score'] * 100 if ent else 0
    logger.debug(f"Entailment score: {score:.2f}")
    return score


def tfidf_bm25_score(question: str, response: str) -> float:
    q_tok = question.lower().split()
    r_tok = response.lower().split()
    bm25 = BM25Okapi([r_tok])
    sc = bm25.get_scores(q_tok)[0] if bm25.get_scores(q_tok) else 0
    score = min(max(sc, 0), 1) * 100
    logger.debug(f"BM25 score: {score:.2f}")
    return score


def slot_match_score(question: str, response: str) -> float:
    qd, rd = nlp(question), nlp(response)
    slots = [t for t in qd if t.dep_ in ("ROOT", "dobj")]
    if not slots:
        return 0.0
    matches = sum(1 for s in slots for t in rd if t.lemma_ == s.lemma_)
    score = matches / len(slots) * 100
    logger.debug(f"Slot match: {score:.2f} ({matches}/{len(slots)})")
    return score


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
    logger.debug("--- score_response start ---")
    logger.debug(f"Question: {question_text}")
    logger.debug(f"Response: {response_text}")

    # Keyword fuzzy match
    doc = nlp(response_text)
    tokens = [
        t.lemma_.lower().strip(string.punctuation)
        for t in doc
        if not t.is_stop and len(t.text) > 2
    ]
    kws = [kw.lower().strip(string.punctuation) for kw in relevant_keywords]
    matched, missing = [], []
    for kw in kws:
        pct = max(fuzz.partial_ratio(kw, tok) for tok in tokens) if tokens else 0
        (matched if pct >= 80 else missing).append(kw)
    keyword_score = len(matched) / len(kws) * 100 if kws else 0
    logger.debug(f"Keyword score: {keyword_score:.2f}")

    # Relevance sub-scores
    sem = semantic_similarity(question_text, response_text)
    ent = entailment_score(question_text, response_text)
    tfidf = tfidf_bm25_score(question_text, response_text)
    slot = slot_match_score(question_text, response_text)

    # Combine into question relevance score
    rel_weights = {"semantic": 0.4, "entailment": 0.3, "tfidf": 0.15, "slot": 0.15}
    question_score = (
        sem * rel_weights["semantic"]
        + ent * rel_weights["entailment"]
        + tfidf * rel_weights["tfidf"]
        + slot * rel_weights["slot"]
    )
    logger.debug(f"Question relev. score: {question_score:.2f}")

    # Hybrid final score
    base_w = {"technical": (0.5, 0.5), "behavioral": (0.3, 0.7), "situational": (0.3, 0.7)}
    kw_w, qt_w = base_w.get(question_type, (0.4, 0.6))
    final_num = deep_round(kw_w * keyword_score + qt_w * question_score, 2)
    score10 = round(final_num / 10, 1)
    logger.debug(f"Final score: {final_num:.2f} -> {score10}/10")

    # Build suggestions
    suggestions: List[Dict[str, str]] = []

    # Scoring Explanation
    suggestions.append({
        "area": "Scoring Explanation",
        "feedback": (
            f"Final score {score10}/10 composed of Keyword Score ({keyword_score:.1f}% * {kw_w}) "
            f"+ Relevance Score ({question_score:.1f}% * {qt_w})."
        )
    })
    # Relevance Breakdown
    suggestions.append({
        "area": "Relevance Breakdown",
        "feedback": (
            f"Semantic: {sem:.1f}%, Entailment: {ent:.1f}%, "
            f"BM25: {tfidf:.1f}%, Slot Match: {slot:.1f}%"
        )
    })
    # Improvement advice per dimension
    if sem < 50:
        suggestions.append({
            "area": "Semantic Alignment",
            "feedback": "Try phrasing your answer using terminology closer to the question prompt."
        })
    if ent < 50:
        suggestions.append({
            "area": "Answer Focus",
            "feedback": "Ensure your response directly addresses the question by explicitly stating your main point early."
        })
    if tfidf < 50:
        suggestions.append({
            "area": "Keyword Inclusion",
            "feedback": "Incorporate more key terms from the question to improve relevance."
        })
    if slot < 50:
        suggestions.append({
            "area": "Slot Coverage",
            "feedback": "Cover the main action and object mentioned in the prompt to better match its intent."
        })
    # Keyword usage advice
    if missing:
        suggestions.append({
            "area": "Keyword Usage",
            "feedback": f"Consider adding missing keywords: {', '.join(missing)}."
        })
    # Structure tips
    if question_type == "behavioral":
        suggestions.append({
            "area": "Behavioral Structure",
            "feedback": "Use STAR (Situation, Task, Action, Result)."
        })
    elif question_type == "situational":
        suggestions.append({
            "area": "Situational Strategy",
            "feedback": "Outline your reasoning steps clearly."
        })
    elif question_type == "technical":
        suggestions.append({
            "area": "Technical Depth",
            "feedback": "Include specific tools or examples."
        })
    if len(response_text.split()) < 20:
        suggestions.append({
            "area": "Detail & Depth",
            "feedback": "Expand with examples or explanations."
        })

    result = {
        "score": score10,
        "keyword_matches": matched,
        "expected_keywords": kws,
        "skills_shown": extract_skills_from_response(response_text),
        "relevance_breakdown": {"semantic": sem, "entailment": ent, "tfidf": tfidf, "slot": slot},
        "improvement_suggestions": suggestions,
        "transcribed_text": response_text
    }
    logger.debug(f"--- score_response result: {result}")
    return result


# --------------------------------------------------
# Debug runner
# --------------------------------------------------
if __name__ == "__main__":
    test_q = "Tell me about a time you led a team successfully."
    test_r = "I led five engineers, improved throughput by 20%, and delivered on time."
    kws, qtype, sec = get_relevant_keywords(
        {"question_text": test_q, "skills": ["leadership"], "experience": ""},
        "PM",
        "Acme",
        "[]"
    )
    import pprint
    pprint.pprint(score_response(test_r, test_q, kws, qtype, sec))
