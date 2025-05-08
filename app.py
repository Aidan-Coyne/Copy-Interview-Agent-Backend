import os
import time                                        # ← NEW
# Set threading to use available CPU cores efficiently
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import process_cv
import search_company
import question_generation
import evaluate_response
import uvicorn
import logging
import json
import tempfile
from typing import Dict

# Import ONNX embedder for caching CV embeddings
from keyword_extraction import onnx_embedder

# Firebase integration
import firebase_admin
from firebase_admin import credentials, storage

print("🔥 FastAPI app starting…")
app = FastAPI()

# ─── Handle FastAPI HTTPExceptions so detail is returned ─────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logging.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "Request error", "detail": exc.detail},
    )

# ─── Catch-all for any other Exception ──────────────────────────────────────────
@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-interview-agent-frontend-production.up.railway.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("CORS Middleware configured.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Firebase setup with error handling
bucket = None
google_creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if google_creds_json:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            temp_file.write(google_creds_json.encode())
            temp_path = temp_file.name
        cred = credentials.Certificate(temp_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'ai-interview-agent-e2f7b.firebasestorage.app'
            })
            logger.info("✅ Firebase initialized.")
        bucket = storage.bucket()
        if not bucket:
            logger.error("❌ Firebase bucket returned None.")
    except Exception as e:
        logger.error(f"❌ Firebase initialization failed: {e}")
else:
    logger.warning("⚠️ GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable not found.")

# ✅ Download ONNX model from Firebase at startup
MODEL_LOCAL_PATH = "/app/models/paraphrase-MiniLM-L3-v2.onnx"
MODEL_FIREBASE_PATH = "models/paraphrase-MiniLM-L3-v2.onnx"

try:
    os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)
    model_blob = bucket.blob(MODEL_FIREBASE_PATH)
    model_blob.download_to_filename(MODEL_LOCAL_PATH)
    logger.info(f"✅ ONNX model downloaded to: {MODEL_LOCAL_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to download ONNX model: {e}")
    raise

# ─── Session Storage ────────────────────────────────────────────────────────────
session_data: Dict[str, Dict] = {}

# ✅ Firebase upload helper
def upload_to_firebase(file_or_bytes, firebase_bucket, path: str, content_type: str) -> str:
    if not firebase_bucket:
        raise HTTPException(status_code=500, detail="Firebase bucket is not available.")
    blob = firebase_bucket.blob(path)
    try:
        if isinstance(file_or_bytes, bytes):
            blob.upload_from_string(file_or_bytes, content_type=content_type)
        else:
            raise HTTPException(status_code=500, detail="Unsupported file type for Firebase upload.")
    except Exception as e:
        logger.error(f"🔥 Firebase upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    blob.make_public()
    return blob.public_url

@app.get("/")
def home():
    return {"message": "Interview Agent Backend Running"}

@app.post("/upload_cv/")
async def upload_cv(
    file: UploadFile = File(...),
    company_name: str = Form(...),
    job_role: str = Form(...),
    question_type: str = Form("mixed")
):
    session_id = f"{company_name}_{job_role}_{file.filename}"
    logger.info(f"Received upload_cv request for session: {session_id}")

    if not file.filename.lower().endswith((".pdf", ".doc", ".docx")):
        raise HTTPException(status_code=400,
                            detail="Unsupported file type. Only PDF and Word documents are allowed.")

    # Upload CV to Firebase
    cv_path = f"sessions/{session_id}/cv/{file.filename}"
    file_bytes = await file.read()
    cv_public_url = upload_to_firebase(file_bytes, bucket, cv_path, file.content_type)
    logger.info(f"✅ CV uploaded to Firebase: {cv_public_url}")

    # Extract CV text and keywords
    cv_text, cv_keywords = process_cv.process_cv_from_firebase(session_id, file.filename, bucket)
    if not cv_text.strip():
        raise HTTPException(status_code=400,
                            detail="CV text extraction failed or CV is empty.")

    # Precompute and cache CV embeddings (ONNX)
    cv_embeddings = onnx_embedder.embed([cv_text])
    session_data[session_id] = {
        "cv_text": cv_text,
        "cv_embeddings": cv_embeddings,
        "cv_keywords": cv_keywords  # ✅ Save CV keywords
    }

    # Fetch company info
    company_info = search_company.search_company_info(company_name, job_role)
    company_info_json = json.loads(company_info)

    # Generate questions and upload audio
    questions_data = question_generation.generate_questions(
        cv_text=cv_text,
        company_info=company_info_json,
        job_role=job_role,
        company_name=company_name,
        selected_question_type=question_type,
        firebase_bucket=bucket,
        session_id=session_id,
        cv_embeddings=cv_embeddings,
        cv_keywords=cv_keywords
    )

    # Store rest of session data
    session_data[session_id].update({
        "company_info": company_info_json,
        "questions": questions_data,
        "job_role": job_role,
        "company_name": company_name
    })

    return {
        "session_id": session_id,
        "cv_text": cv_text,
        "company_info": company_info_json,
        "questions_data": questions_data,
        "cv_url": cv_public_url
    }

@app.post("/evaluate_response/")
async def evaluate_audio_response(
    audio_file: UploadFile = File(...),
    question_index: int = Form(...),
    session_id: str = Form(...)
):
    if session_id not in session_data:
        raise HTTPException(status_code=400,
                            detail="Session not found. Please generate questions first.")

    total_start = time.time()
    logger.info(f"⏳ [evaluate_response] start session={session_id!r} question_index={question_index}")

    session = session_data[session_id]
    questions = session.get("questions", [])
    if question_index >= len(questions):
        raise HTTPException(status_code=400, detail="Invalid question index.")

    question_data = questions[question_index]

    # ─── read body ───────────────────────────────────────────────────────────
    read_start = time.time()
    audio_data = await audio_file.read()
    logger.info(f"⏱ read multipart in {(time.time() - read_start):.2f}s")

    # ─── Upload response ─────────────────────────────────────────────────────
    upload_start = time.time()
    response_path = f"sessions/{session_id}/audio_responses/response_{question_index + 1}.webm"
    response_url  = upload_to_firebase(audio_data, bucket, response_path, "audio/webm")
    logger.info(f"⏱ firebase.upload took {(time.time() - upload_start):.2f}s")

    # ─── Convert & transcribe ───────────────────────────────────────────────
    convert_start = time.time()
    wav_data      = evaluate_response.convert_to_wav(audio_data)
    response_text = evaluate_response.transcribe_audio(wav_data)
    logger.info(f"⏱ convert+transcribe took {(time.time() - convert_start):.2f}s")

    # ─── Evaluate ───────────────────────────────────────────────────────────
    score_start = time.time()
    relevant_keywords, question_type, company_sector = \
        evaluate_response.get_relevant_keywords(
            question_data,
            session["job_role"],
            session["company_name"],
            session["company_info"]
        )
    result = evaluate_response.score_response(
        response_text,
        question_data["question_text"],
        relevant_keywords,
        question_type,
        company_sector,
        wav_bytes=wav_data             # ← pass the WAV bytes here
    )
    logger.info(f"⏱ scoring pipeline took {(time.time() - score_start):.2f}s")

    total_elapsed = time.time() - total_start
    logger.info(f"✅ [evaluate_response] total time: {total_elapsed:.2f}s")

    return JSONResponse(
        content={
            "feedback": result,
            "transcribed_text": response_text,
            "response_audio_url": response_url
        }
    )

@app.post("/end_session/{session_id}")
def end_session(session_id: str):
    if session_id in session_data:
        del session_data[session_id]
        logger.info(f"Session {session_id} data removed from session_data.")
    return {"message": f"Session {session_id} ended successfully."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
