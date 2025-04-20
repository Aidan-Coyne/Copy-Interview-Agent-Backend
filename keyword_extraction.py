import os
import re
import logging
import time
import numpy as np
import onnxruntime as rt
from transformers import AutoTokenizer
from keybert import KeyBERT
from keybert.backend._base import BaseEmbedder

# Persist transformers cache in /tmp across container lifetime
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"

# ─── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Load tokenizer & ONNX session at startup ──────────────────────────────────
_model_start = time.time()

# Adjust path to wherever your ONNX file lives in the container
ONNX_PATH   = "/app/models/paraphrase-MiniLM-L3-v2.onnx"
TOKENIZER   = AutoTokenizer.from_pretrained("paraphrase-MiniLM-L3-v2")
SESSION     = rt.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

logger.info(f"Loaded ONNX model in {time.time() - _model_start:.2f}s")

# ─── ONNX embedder for KeyBERT ──────────────────────────────────────────────────
class ONNXEmbedder(BaseEmbedder):
    def __init__(self, session: rt.InferenceSession, tokenizer: AutoTokenizer):
        self.sess      = session
        self.tokenizer = tokenizer
        # Map ONNX input/output names
        inputs        = {inp.name: inp.name for inp in session.get_inputs()}
        self.input_ids      = inputs.get("input_ids", list(inputs)[0])
        self.attention_mask = inputs.get("attention_mask", list(inputs)[1])
        self.output_name    = session.get_outputs()[0].name

    def embed(self, documents: list[str]) -> np.ndarray:
        # Tokenize + convert to numpy arrays
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
        # Run ONNX Runtime
        embeddings = self.sess.run([self.output_name], ort_inputs)[0]
        return embeddings

# Instantiate KeyBERT with ONNX backend
onnx_embedder = ONNXEmbedder(SESSION, TOKENIZER)
kw_model      = KeyBERT(model=onnx_embedder)


# ─── Noise‑word filtering ───────────────────────────────────────────────────────
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
    top_n: int    = 10,
    min_len: int  = 3,
    embeddings: np.ndarray | None = None
) -> list[str]:
    """
    Extracts keywords using KeyBERT + ONNX embeddings.

    Args:
      text (str): full document text
      top_n (int): how many keywords to return
      embeddings (np.ndarray|null):
          If provided, KeyBERT will reuse these (shape must match [1, dim]).
    Returns:
      List of cleaned keywords.
    """
    if not text or len(text) < 50:
        logger.warning("Input text too short for reliable extraction.")
        return []

    start = time.time()
    try:
        raw = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            use_mmr=True,
            nr_candidates=20,
            top_n=top_n,
            embeddings=embeddings
        )
        logger.info(f"Raw KeyBERT candidates: {raw}")

        # Clean & filter
        cleaned = []
        for item in raw:
            kw = item[0] if isinstance(item, tuple) else item
            if len(kw) >= min_len and clean_keyword(kw):
                cleaned.append(kw)

        logger.info(f"Extracted & cleaned keywords: {cleaned}")
        logger.info(f"⏱ Keyword extraction took {time.time() - start:.2f}s")
        return cleaned

    except Exception as e:
        logger.error(f"Error during keyword extraction: {e}", exc_info=True)
        return []
