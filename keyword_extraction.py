import re
import logging
from keybert import KeyBERT

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the KeyBERT model
kw_model = KeyBERT()

# Define a set of common noise words or terms to ignore
STOPWORDS = set([
    "linkedin", "profile", "cv", "resume", "email", "phone", "contact",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"
])

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
    if not text or len(text) < 50:
        logger.warning("Input text is too short for reliable keyword extraction.")
        return []

    try:
        raw_keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=top_n)
        logger.info(f"Raw keywords from KeyBERT: {raw_keywords}")

        cleaned_keywords = []
        for item in raw_keywords:
            if isinstance(item, tuple) and len(item) == 2:
                kw = item[0]
            elif isinstance(item, str):
                kw = item
            else:
                continue  # Skip unexpected structures

            if len(kw) >= min_len and clean_keyword(kw):
                cleaned_keywords.append(kw)

        logger.info(f"Extracted and cleaned keywords: {cleaned_keywords}")
        return cleaned_keywords

    except Exception as e:
        logger.error(f"Error during keyword extraction: {e}")
        return []
