import os
import re
import logging
import time
import json

import numpy as np
import onnxruntime as rt
from transformers import AutoTokenizer
from keybert import KeyBERT
from keybert.backend._base import BaseEmbedder
import spacy

import firebase_admin
from firebase_admin import credentials, storage

# ─── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Firebase initialization ───────────────────────────────────────────────────
creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", None)
if not creds_json:
    raise RuntimeError("Set GOOGLE_APPLICATION_CREDENTIALS_JSON to your service account JSON")

creds_dict = json.loads(creds_json)
if not firebase_admin._apps:
    cred = credentials.Certificate(creds_dict)
    firebase_admin.initialize_app(cred, {
        "storageBucket": "ai-interview-agent-e2f7b.firebasestorage.app"
    })
bucket = storage.bucket()

# ─── Prepare local models folder ────────────────────────────────────────────────
HERE        = os.path.dirname(__file__)
MODELS_DIR  = os.path.join(HERE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

REMOTE_ONNX_PATH = "models/paraphrase-MiniLM-L3-v2.onnx"
LOCAL_ONNX_PATH  = os.path.join(MODELS_DIR, "paraphrase-MiniLM-L3-v2.onnx")

# ─── Download ONNX model from Firebase if missing ───────────────────────────────
if not os.path.exists(LOCAL_ONNX_PATH):
    logger.info("Downloading ONNX model from Firebase…")
    blob = bucket.blob(REMOTE_ONNX_PATH)
    blob.download_to_filename(LOCAL_ONNX_PATH)
    logger.info("✅ Download complete.")

# ─── Load tokenizer & ONNX session at startup ──────────────────────────────────
_model_start = time.time()
TOKENIZER = AutoTokenizer.from_pretrained(
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN", None)
)
SESSION = rt.InferenceSession(LOCAL_ONNX_PATH, providers=["CPUExecutionProvider"])
logger.info(f"Loaded ONNX model in {time.time() - _model_start:.2f}s")

# ─── ONNX embedder for KeyBERT ──────────────────────────────────────────────────
class ONNXEmbedder(BaseEmbedder):
    def __init__(self, session: rt.InferenceSession, tokenizer: AutoTokenizer):
        self.sess      = session
        self.tokenizer = tokenizer
        inputs         = {inp.name: inp.name for inp in session.get_inputs()}
        self.input_ids      = inputs.get("input_ids", list(inputs)[0])
        self.attention_mask = inputs.get("attention_mask", list(inputs)[1])
        self.output_name    = session.get_outputs()[0].name

    def embed(self, documents: list[str] | np.ndarray) -> np.ndarray:
        if isinstance(documents, np.ndarray):
            return documents
        enc = self.tokenizer(
            documents,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512
        )
        ort_inputs = {
            self.input_ids:      enc["input_ids"],
            self.attention_mask: enc["attention_mask"]
        }
        embeddings = self.sess.run([self.output_name], ort_inputs)[0]
        return embeddings

# Instantiate KeyBERT with ONNX backend
onnx_embedder = ONNXEmbedder(SESSION, TOKENIZER)
kw_model      = KeyBERT(model=onnx_embedder)

# ─── Additional NLP filtering (spaCy) ───────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")

STOPWORDS = {
    "linkedin", "profile", "cv", "resume", "email", "phone", "contact", "name", "gmail",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"
}

def is_clean_keyword(keyword: str) -> bool:
    if not keyword or len(keyword) < 2:
        return False
    if any(symbol in keyword for symbol in ["@", "+", "/", "\\"]):
        return False
    if keyword.lower() in STOPWORDS:
        return False
    if re.search(r"\b\d{2,}\b", keyword):
        return False
    return True

def get_role_whitelist(job_role: str, job_sector: str, job_role_library: dict) -> set[str]:
    sector_data = job_role_library.get(job_sector, {})
    role_keywords = set(sector_data.get(job_role, []))
    sector_keywords = set(sector_data.get("_keywords", []))
    return {kw.lower() for kw in (role_keywords | sector_keywords)}

def is_skill_like(keyword: str, whitelist: set[str]) -> bool:
    if keyword.lower() in whitelist:
        return True
    doc = nlp(keyword)
    return any(tok.pos_ in {"NOUN", "PROPN"} for tok in doc)

# ─── Main extraction function ───────────────────────────────────────────────────
def extract_keywords(
    text: str,
    job_role: str,
    job_sector: str,
    job_role_library: dict,
    top_n: int = 10
) -> list[str]:
    try:
        if text.strip().startswith("{") or text.strip().startswith("["):
            try:
                data = json.loads(text)
                if isinstance(data, list) and data and isinstance(data[0], dict) and "snippet" in data[0]:
                    text = " ".join(
                        item["snippet"]
                        for item in data
                        if isinstance(item.get("snippet"), str)
                    )
            except json.JSONDecodeError:
                pass

        if not text or len(text) < 50:
            logger.warning("⚠️ Input text too short for reliable extraction.")
            return []

        role_whitelist = get_role_whitelist(job_role, job_sector, job_role_library)

        raw = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            use_mmr=True,
            nr_candidates=30,
            top_n=top_n * 2
        )

        final_keywords = []
        for kw, _ in raw:
            kw = kw.strip().lower()
            if not is_clean_keyword(kw):
                continue
            if not is_skill_like(kw, role_whitelist):
                continue
            final_keywords.append(kw)
            if len(final_keywords) >= top_n:
                break

        logger.info(f"✅ Final extracted keywords: {final_keywords}")
        return final_keywords

    except Exception as e:
        logger.error(f"❌ Error during keyword extraction: {e}", exc_info=True)
        return []