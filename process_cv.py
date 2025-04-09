import pdfplumber
from fastapi import UploadFile, HTTPException
import logging
from docx import Document
from keyword_extraction import extract_keywords

logging.basicConfig(level=logging.INFO)

def extract_text_from_pdf(file: UploadFile) -> str:
    """
    Extracts text from a PDF file.
    """
    try:
        with pdfplumber.open(file.file) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        text = " ".join(text.split())
        return text
    except pdfplumber.PDFError:
        logging.error("Invalid PDF file uploaded.")
        raise HTTPException(status_code=400, detail="Invalid PDF file.")
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting text: {e}")

def extract_text_from_docx(file: UploadFile) -> str:
    """
    Extracts text from a Word document (.docx or .doc).
    """
    try:
        doc = Document(file.file)
        text = "\n".join(para.text for para in doc.paragraphs if para.text)
        text = " ".join(text.split())
        return text
    except Exception as e:
        logging.error(f"Error extracting text from DOCX: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting text: {e}")

def extract_text_from_file(file: UploadFile) -> str:
    """
    Determines file type and extracts text accordingly.
    Supports PDF and Word documents.
    """
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif filename.endswith(".docx") or filename.endswith(".doc"):
        return extract_text_from_docx(file)
    else:
        logging.error("Unsupported file type uploaded.")
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and Word documents are allowed.")

def extract_keywords_from_cv(file: UploadFile, top_n: int = 10) -> list:
    """
    Extracts text from the file and returns a list of keywords using the keyword_extraction module.
    """
    text = extract_text_from_file(file)
    keywords = extract_keywords(text, top_n=top_n)
    return keywords

def process_cv(file: UploadFile, top_n: int = 10) -> tuple:
    """
    Processes a CV file (PDF or Word), returning the extracted text and keywords.
    """
    text = extract_text_from_file(file)
    keywords = extract_keywords(text, top_n=top_n)
    return text, keywords
