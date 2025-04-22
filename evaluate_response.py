import speech_recognition as sr
import spacy
from fuzzywuzzy import fuzz
import json
import logging
from typing import List, Dict, Tuple, Any, Optional
from fastapi import HTTPException
from io import BytesIO
import wave
import subprocess, uuid, os, tempfile, string
from google.cloud import storage

# Optional: install rank-bm25 via pip install rank-bm25
from rank_bm25 import BM25Okapi

# NEW imports for semantic relevance
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from question_generation import extract_sectors

# Logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Helper rounding function
def deep_round(value: float, ndigits: int) -> float:
    """Round a float to a given precision."""
    return round(value, ndigits)

# Load models once at import time
envb = os.environ.get
embedder = SentenceTransformer('all-MiniLM-L6-v2')
nli = pipeline("text-classification", model="roberta-large-mnli")
nlp = spacy.load("en_core_web_sm")

class TranscriptionError(Exception):
    pass

# ------------------------------------------------------------------------------------------------
# Core utility functions
# ------------------------------------------------------------------------------------------------

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
        subprocess.run(["ffmpeg", "-y", "-i", input_filename, "-acodec", "pcm_s16le", output_filename], check=True)
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
            kw.extend(token.lemma_ for token in doc if token.pos_ in {"NOUN","VERB"})
            kw.extend(ent.text.lower() for ent in doc.ents)
        return list(set(kw))
    except Exception as e:
        logger.error(f"Error extracting company keywords: {e}")
        return []


def extract_skills_from_response(response_text: str) -> List[str]:
    doc = nlp(response_text)
    return list(set(token.text.lower() for token in doc if token.pos_ in {"NOUN","VERB"}))


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
    suggestions: List[Dict[str, str]] = []
    suggestions.append({
        "area": "Scoring Explanation",
        "feedback": (
            f"Your final score is {final_score} out of 10. Calculated as: "
            f"(Keyword: {keyword_score:.1f}% * {weights.get('keyword',0)}) + "
            f"(Question: {question_score:.1f}% * {weights.get('question',0)}) = "
            f"{(keyword_score*weights.get('keyword',0)+question_score*weights.get('question',0)):.1f}%"
        )
    })
    if missing_keywords:
        suggestions.append({
            "area": "Keyword Usage",
            "feedback": f"Consider adding missing keywords: {', '.join(missing_keywords)}"
        })
    if question_type == "behavioral":
        suggestions.append({"area":"Behavioral Structure","feedback":"Use the STAR method."})
    elif question_type == "situational":
        suggestions.append({"area":"Situational Strategy","feedback":"Outline step-by-step reasoning."})
    elif question_type == "technical":
        suggestions.append({"area":"Technical Depth","feedback":"Include specific tools/examples."})
    if len(response_text.split()) < 20:
        suggestions.append({"area":"Detail & Depth","feedback":"Expand with more examples."})
    return suggestions


def get_relevant_keywords(
    question_data: Dict[str, Any],
    job_role: str,
    company_name: str,
    company_info: str
) -> Tuple[List[str], str, Optional[str]]:
    q_skills = question_data.get("skills", [])
    q_exp = question_data.get("experience", "")
    q_text = question_data.get("question_text", "").lower()
    comp_kw = extract_company_keywords(company_info)
    role_lemma = nlp(job_role)[0].lemma_ if job_role else ""
    name_lemma = nlp(company_name)[0].lemma_ if company_name else ""
    exp_lemma = nlp(q_exp)[0].lemma_ if q_exp else ""
    all_kw = set([nlp(s)[0].lemma_.lower() for s in q_skills]+([exp_lemma] if exp_lemma else[])+comp_kw+[role_lemma,name_lemma])
    question_type = "general"
    if any(k in q_text for k in ["tell me about a time","describe a situation"]):
        question_type = "behavioral"
    elif "how would you" in q_text:
        question_type = "situational"
    elif any(k in q_text for k in ["skills","explain"]):
        question_type = "technical"
    try:
        sector = extract_sectors(json.loads(company_info))
    except Exception:
        sector = None
    return list(all_kw), question_type, sector

# ------------------------------------------------------------------------------------------------
# Relevance model functions
# ------------------------------------------------------------------------------------------------

def semantic_similarity(q: str, r: str) -> float:
    q_emb = embedder.encode(q, convert_to_tensor=True)
    r_emb = embedder.encode(r, convert_to_tensor=True)
    score = (util.cos_sim(q_emb, r_emb).item() + 1) / 2 * 100
    logger.debug(f"Semantic similarity: {score:.2f}")
    return score


def entailment_score(question: str, response: str) -> float:
    out = nli(f"{question} </s></s> {response}")
    ent = next((x for x in out if x['label']=='ENTAILMENT'), None)
    score = ent['score']*100 if ent else 0
    logger.debug(f"Entailment score: {score:.2f}")
    return score


def tfidf_bm25_score(question: str, response: str) -> float:
    q_tok = question.lower().split()
    r_tok = response.lower().split()
    bm25 = BM25Okapi([r_tok])
    sc = bm25.get_scores(q_tok)[0] if bm25.get_scores(q_tok) else 0
    score = min(max(sc,0),1)*100
    logger.debug(f"BM25 score: {score:.2f}")
    return score


def slot_match_score(question: str, response: str) -> float:
    qd, rd = nlp(question), nlp(response)
    slots = [t for t in qd if t.dep_ in ("ROOT","dobj")]
    if not slots:
        return 0.0
    matches = sum(1 for s in slots for t in rd if t.lemma_==s.lemma_)
    score = matches/len(slots)*100
    logger.debug(f"Slot match: {score:.2f} ({matches}/{len(slots)})")
    return score

# ------------------------------------------------------------------------------------------------
# Main scoring function
# ------------------------------------------------------------------------------------------------

def score_response(
    response_text: str,
    question_text: str,
    relevant_keywords: List[str],
    question_type: str,
    company_sector: Optional[str]=None
) -> Dict[str, Any]:
    logger.debug("--- score_response start ---")
    logger.debug(f"Question: {question_text}")
    logger.debug(f"Response: {response_text}")

    # Keyword fuzzy match
    doc = nlp(response_text)
    tokens = [t.lemma_.lower().strip(string.punctuation) for t in doc if not t.is_stop and len(t)>2]
    kws = [kw.lower().strip(string.punctuation) for kw in relevant_keywords]
    matched, missing = [], []
    for kw in kws:
        pct = max(fuzz.partial_ratio(kw, tok) for tok in tokens)
        (matched if pct>=80 else missing).append(kw)
    keyword_score = len(matched)/len(kws)*100 if kws else 0
    logger.debug(f"Keyword score: {keyword_score:.2f}")

    # Relevance sub-scores
    sem = semantic_similarity(question_text, response_text)
    ent = entailment_score(question_text, response_text)
    tfidf = tfidf_bm25_score(question_text, response_text)
    slot = slot_match_score(question_text, response_text)

    rel_weights = {"semantic":0.4,"entailment":0.3,"tfidf":0.15,"slot":0.15}
    question_score = sem*rel_weights['semantic']+ent*rel_weights['entailment']+tfidf*rel_weights['tfidf']+slot*rel_weights['slot']
    logger.debug(f"Question relev. score: {question_score:.2f}")

    base_w = {"technical":(0.5,0.5),"behavioral":(0.3,0.7),"situational":(0.3,0.7)}
    kw_w, qt_w = base_w.get(question_type,(0.4,0.6))
    final_num = deep_round(kw_w*keyword_score + qt_w*question_score,2)
    score10 = round(final_num/10,1)
    logger.debug(f"Final score: {final_num:.2f} -> {score10}/10")

    suggestions = generate_improvement_suggestions(response_text, missing, question_type, kws, score10, keyword_score, question_score, {"keyword":kw_w,"question":qt_w})
    if company_sector:
        suggestions.append({"area":"Sector Fit","feedback":f"Your answer shows awareness of the {company_sector} sector."})

    result = {
        "score": score10,
        "keyword_matches": matched,
        "expected_keywords": kws,
        "skills_shown": extract_skills_from_response(response_text),
        "relevance_breakdown": {"semantic":sem,"entailment":ent,"tfidf":tfidf,"slot":slot},
        "improvement_suggestions": suggestions,
        "transcribed_text": response_text
    }
    logger.debug(f"--- score_response end: {result} ---")
    return result

# ------------------------------------------------------------------------------------------------
# Debug runner
# ------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    test_q = "Tell me about a time you led a team successfully."
    test_r = "I led five engineers, improved throughput by 20%, and delivered on time."
    kws, qtype, sec = get_relevant_keywords({"skills":["leadership"]},"PM","Acme","[]")
    import pprint; pprint.pprint(score_response(test_r, test_q, kws, qtype, sec))
