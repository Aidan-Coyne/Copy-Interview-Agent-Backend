import pdfplumber
from fastapi import UploadFile, HTTPException
import logging
from docx import Document
from keyword_extraction import extract_keywords
from google.cloud import storage
from job_role_library import job_role_library
import tempfile
import os
import time

logging.basicConfig(level=logging.INFO)

# 🔥 Download CV from Firebase
def download_cv_from_firebase(session_id: str, filename: str, bucket: storage.Bucket) -> str:
    blob_path = f"sessions/{session_id}/cv/{filename}"
    blob = bucket.blob(blob_path)

    if not blob.exists():
        logging.error(f"CV file not found in Firebase: {blob_path}")
        raise HTTPException(status_code=404, detail="CV not found in storage.")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
    blob.download_to_filename(temp_file.name)
    logging.info(f"📥 Downloaded CV from Firebase to: {temp_file.name}")
    return temp_file.name

def extract_text_from_pdf_path(file_path: str) -> str:
    logging.info(f"🔍 Attempting to extract text from PDF: {file_path}")
    try:
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        cleaned_text = " ".join(text.split())
        logging.info(f"✅ Extracted PDF text length: {len(cleaned_text)} characters")
        return cleaned_text
    except Exception as e:
        logging.error(f"❌ Error extracting text from PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {e}")

def extract_text_from_docx_path(file_path: str) -> str:
    logging.info(f"🔍 Attempting to extract text from DOCX: {file_path}")
    try:
        doc = Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs if para.text)
        cleaned_text = " ".join(text.split())
        logging.info(f"✅ Extracted DOCX text length: {len(cleaned_text)} characters")
        return cleaned_text
    except Exception as e:
        logging.error(f"❌ Error extracting text from DOCX: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting text from DOCX: {e}")

def extract_text_from_file_path(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf_path(file_path)
    elif file_path.lower().endswith((".doc", ".docx")):
        return extract_text_from_docx_path(file_path)
    else:
        logging.error("❌ Unsupported file type in Firebase download.")
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and Word documents are allowed.")

# 🎯 Extract relevant experience from keyword match against job role
def extract_relevant_experience_from_keywords(cv_keywords: list[str], job_role: str) -> str:
    job_role_lower = job_role.lower()
    for sector, roles in job_role_library.items():
        for role, keywords in roles.items():
            if role.lower() == job_role_lower and isinstance(keywords, list):
                matched = [kw for kw in cv_keywords if kw in keywords]
                return ", ".join(matched[:2]) if matched else "your field"
    return "your field"

# ✅ Main CV processing entry point
def process_cv_from_firebase(session_id: str, filename: str, bucket: storage.Bucket, job_role: str = None, top_n: int = 10) -> tuple:
    start = time.time()

    logging.info("⏱ Starting CV processing pipeline...")
    local_path = download_cv_from_firebase(session_id, filename, bucket)
    logging.info(f"⏱ Firebase download time: {time.time() - start:.2f}s")

    text_start = time.time()
    text = extract_text_from_file_path(local_path)
    logging.info(f"⏱ Text extraction time: {time.time() - text_start:.2f}s")

    keyword_start = time.time()
    logging.info("🔑 Extracting keywords from CV text...")
    keywords = extract_keywords(text, top_n=top_n)
    logging.info(f"✅ Extracted {len(keywords)} keywords")
    logging.info(f"⏱ Keyword extraction time: {time.time() - keyword_start:.2f}s")

    relevant_experience = None
    if job_role:
        relevant_experience = extract_relevant_experience_from_keywords(keywords, job_role)
        logging.info(f"🎯 Extracted relevant experience: {relevant_experience}")

    try:
        os.remove(local_path)
        logging.info(f"🧹 Temp file deleted: {local_path}")
    except Exception as e:
        logging.warning(f"⚠️ Failed to delete temp CV file: {e}")

    total_time = time.time() - start
    logging.info(f"🎯 Total CV processing time: {total_time:.2f}s")

    return text, keywords, relevant_experience
