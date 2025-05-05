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

# ─── Noise-word filtering ───────────────────────────────────────────────────────
STOPWORDS = {
    "linkedin", "profile", "cv", "resume", "email", "phone", "contact",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"
}

def clean_keyword(keyword: str) -> bool:
    if keyword.isdigit() or len(keyword) < 3:
        return False
    if re.search(r"[\d]{2,}", keyword) or "+" in keyword or "@" in keyword:
        return False
    if keyword.lower() in STOPWORDS:
        return False
    return True

# ─── Main extraction function ─────────────────────────────────────────────────
def extract_keywords(
    text: str,
    top_n: int   = 10,
    min_len: int = 3
) -> list[str]:
    try:
        data = json.loads(text)
        if (
            isinstance(data, list) and data and
            isinstance(data[0], dict) and "snippet" in data[0]
        ):
            text_input = " ".join(
                item["snippet"]
                for item in data
                if isinstance(item.get("snippet"), str)
            )
        else:
            text_input = text
    except json.JSONDecodeError:
        text_input = text

    if not text_input or len(text_input) < 50:
        logger.warning("Input text too short for reliable extraction.")
        return []

    start = time.time()
    try:
        raw = kw_model.extract_keywords(
            text_input,
            keyphrase_ngram_range=(1, 2),
            use_mmr=True,
            nr_candidates=20,
            top_n=top_n
        )
        logger.info(f"Raw KeyBERT candidates: {raw}")

        final_keywords = []
        for kw, _ in raw:
            kw = kw.strip().lower()
            if not clean_keyword(kw):
                continue
            if " " in kw:
                # Only discard if both parts are bad
                parts = kw.split()
                if all(not clean_keyword(p) for p in parts):
                    continue
            final_keywords.append(kw)

        logger.info(f"Extracted & cleaned keywords: {final_keywords}")
        logger.info(f"⏱ Keyword extraction took {time.time() - start:.2f}s")
        return final_keywords[:top_n]

    except Exception as e:
        logger.error(f"Error during keyword extraction: {e}", exc_info=True)
        return []
