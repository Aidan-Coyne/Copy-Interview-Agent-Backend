import os
import re
import logging
import time

# Ensure HF transformers cache is in /tmp (persisted during container lifetime)
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Preload embedding & KeyBERT models ─────────────────────────────────────────
_model_start = time.time()
# Use a lighter, faster model
embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
kw_model = KeyBERT(model=embedding_model)
logger.info(f"⏱ Loaded embedding & KeyBERT models in {time.time() - _model_start:.2f}s")
# ────────────────────────────────────────────────────────────────────────────────

# Define a set of common noise words or terms to ignore
STOPWORDS = {
    "linkedin", "profile", "cv", "resume", "email", "phone", "contact",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"
}

def clean_keyword(keyword: str) -> bool:
    """
    Determines whether a keyword is valid:
    - Ignores numeric-only terms or those too short.
    - Excludes contact-like patterns.
    - Excludes common stopwords.
    """
    if keyword.isdigit() or len(keyword) < 3:
        return False
    if re.search(r"[\d]{2,}", keyword) or "+" in keyword or "@" in keyword:
        return False
    if keyword.lower() in STOPWORDS:
        return False
    return True

def extract_keywords(text: str, top_n: int = 10, min_len: int = 3) -> list:
    """
    Extracts a list of keywords from the input text using KeyBERT,
    and then cleans the results by filtering out noisy or irrelevant keywords.

    Args:
        text (str): The text from which to extract keywords.
        top_n (int): Maximum number of keywords to extract (default is 10).
        min_len (int): Minimum length for a valid keyword (default is 3).

    Returns:
        list: A list of extracted and cleaned keywords.
    """
    start = time.time()
    if not text or len(text) < 50:
        logger.warning("Input text is too short for reliable keyword extraction.")
        return []

    try:
        raw = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            use_mmr=True,             # Maximal marginal relevance for more diverse
            nr_candidates=20,         # generate more candidates and then pick top_n
            top_n=top_n
        )
        logger.info(f"Raw KeyBERT candidates: {raw}")

        cleaned = []
        for item in raw:
            kw = item[0] if isinstance(item, tuple) else item
            if len(kw) >= min_len and clean_keyword(kw):
                cleaned.append(kw)

        duration = time.time() - start
        logger.info(f"Extracted & cleaned keywords: {cleaned}")
        logger.info(f"⏱ Keyword extraction took {duration:.2f}s")
        return cleaned

    except Exception as e:
        logger.error(f"❌ Error during keyword extraction: {e}")
        return []
