from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import process_cv
import search_company
import question_generation
import evaluate_response
import uvicorn
import logging
import json
import os
import shutil
import tempfile
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware

# Firebase integration
import firebase_admin
from firebase_admin import credentials, storage

app = FastAPI()

# ✅ CORS: Allow only your frontend origin
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

# ✅ Firebase credentials from Railway environment
google_creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
bucket = None

if google_creds_json:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_file.write(google_creds_json.encode())
        temp_path = temp_file.name
    cred = credentials.Certificate(temp_path)

    # ✅ Initialize only if not already initialized
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'ai-interview-agent-e2f7b.appspot.com'
        })
        bucket = storage.bucket()

# Upload folders
UPLOAD_FOLDER = "uploaded_cvs"
USER_RESPONSES_FOLDER = "user_responses"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(USER_RESPONSES_FOLDER, exist_ok=True)
app.mount("/user_responses", StaticFiles(directory=USER_RESPONSES_FOLDER), name="user_responses")

session_data: Dict[str, Dict] = {}

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
    logger.info(f"Received upload_cv request for company: {company_name}, role: {job_role}, file: {file.filename}")
    try:
        filename_lower = file.filename.lower()
        if not (filename_lower.endswith(".pdf") or filename_lower.endswith(".doc") or filename_lower.endswith(".docx")):
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and Word documents are allowed.")

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        cv_text = process_cv.extract_text_from_file(file)
        if not cv_text.strip():
            raise HTTPException(status_code=400, detail="CV text extraction failed or CV is empty.")

        company_info = search_company.search_company_info(company_name, job_role)
        company_info_json = json.loads(company_info)

        questions_data = question_generation.generate_questions(
            cv_text, company_info_json, job_role, company_name,
            selected_question_type=question_type,
            firebase_bucket=bucket
        )

        session_id = f"{company_name}_{job_role}_{file.filename}"
        session_data[session_id] = {
            "cv_text": cv_text,
            "company_info": company_info_json,
            "questions": questions_data,
            "job_role": job_role,
            "company_name": company_name
        }

        return {
            "session_id": session_id,
            "cv_text": cv_text,
            "company_info": company_info_json,
            "questions_data": questions_data
        }

    except HTTPException as http_exception:
        logger.error(f"HTTP Exception: {http_exception.detail}")
        raise http_exception
    except Exception as e:
        logger.exception("Unexpected error in upload_cv endpoint.")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.post("/evaluate_response/")
async def evaluate_audio_response(
    audio_file: UploadFile = File(...),
    question_index: int = Form(...),
    session_id: str = Form(...)
):
    try:
        if session_id not in session_data:
            raise HTTPException(status_code=400, detail="Session not found. Please generate questions first.")

        session = session_data[session_id]
        questions = session["questions"]

        if question_index >= len(questions):
            raise HTTPException(status_code=400, detail="Invalid question index.")

        question_data = questions[question_index]
        audio_data = await audio_file.read()

        wav_data = evaluate_response.convert_to_wav(audio_data)
        response_text = evaluate_response.transcribe_audio(wav_data)

        relevant_keywords, question_type, company_sector = evaluate_response.get_relevant_keywords(
            question_data, session["job_role"], session["company_name"], json.dumps(session["company_info"])
        )

        result = evaluate_response.score_response(
            response_text, question_data["question_text"], relevant_keywords, question_type, company_sector
        )

        return JSONResponse(content={"feedback": result})

    except evaluate_response.TranscriptionError as e:
        raise HTTPException(status_code=400, detail=f"Audio transcription failed: {e}")
    except HTTPException as http_exception:
        raise http_exception
    except Exception as e:
        logger.exception("Unexpected error in evaluate_response endpoint.")
        raise HTTPException(status_code=500, detail="An error occurred while evaluating the response.")

@app.post("/end_session/{session_id}")
def end_session(session_id: str):
    if session_id in session_data:
        del session_data[session_id]
        logger.info(f"Session {session_id} data removed from session_data.")
    return {"message": f"Session {session_id} ended successfully."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
