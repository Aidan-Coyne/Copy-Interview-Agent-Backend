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
import tempfile  # <-- Added for GCP credential patch
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("CORS Middleware configured.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Google credentials patch for Railway
google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if google_creds and google_creds.strip().startswith('{'):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_file.write(google_creds.encode())
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file.name

UPLOAD_FOLDER = "uploaded_cvs"
AUDIO_FOLDER = "british_audio_questions"
USER_RESPONSES_FOLDER = "user_responses"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(USER_RESPONSES_FOLDER, exist_ok=True)

app.mount("/british_audio_questions", StaticFiles(directory=AUDIO_FOLDER), name="british_audio_questions")
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
    question_type: str = Form("mixed")  # New parameter to select question type
):
    logger.info(f"Received upload_cv request for company: {company_name}, role: {job_role}, file: {file.filename}")
    try:
        filename_lower = file.filename.lower()

        # Validate file type
        if not (filename_lower.endswith(".pdf") or filename_lower.endswith(".doc") or filename_lower.endswith(".docx")):
            logger.error(f"Unsupported file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and Word documents are allowed.")

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.debug(f"File saved to {file_path}")

        # Extract text from CV
        try:
            cv_text = process_cv.extract_text_from_file(file)
            logger.debug(f"Extracted CV text length: {len(cv_text)}")
        except Exception as e:
            logger.exception("Error extracting text from CV.")
            raise HTTPException(status_code=500, detail=f"Error extracting CV text: {str(e)}")

        if not cv_text.strip():
            logger.error("CV text extraction failed or is empty.")
            raise HTTPException(status_code=400, detail="CV text extraction failed or CV is empty.")

        # Fetch company information
        try:
            company_info = search_company.search_company_info(company_name, job_role)
            logger.debug(f"Raw company_info from Brave API: {company_info}")
            company_info_json = json.loads(company_info)
        except Exception as e:
            logger.exception("Error fetching or parsing company info.")
            raise HTTPException(status_code=500, detail=f"Error fetching company info: {str(e)}")

        logger.info(f"Company Info Retrieved: {company_info_json}")
        logger.info(f"Generating {question_type} questions for {company_name} - {job_role}")

        # Generate interview questions with the selected question type
        try:
            questions_data = question_generation.generate_questions(
                cv_text, company_info_json, job_role, company_name, selected_question_type=question_type
            )
            logger.debug(f"Questions Data Generated: {questions_data}")
        except Exception as e:
            logger.exception("Error in question_generation.generate_questions.")
            raise HTTPException(status_code=500, detail=f"Error generating interview questions: {str(e)}")

        base_url = "http://127.0.0.1:8000/british_audio_questions/"
        for question in questions_data:
            if "audio_file" in question and isinstance(question["audio_file"], str):
                question["audio_file"] = base_url + os.path.basename(question["audio_file"])
            else:
                logger.error(f"Invalid audio file reference: {question}")

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
        logger.info(f"Received evaluation request for session_id: {session_id}, question_index: {question_index}")

        if session_id not in session_data:
            raise HTTPException(status_code=400, detail="Session not found. Please generate questions first.")

        session = session_data[session_id]
        questions = session["questions"]

        if question_index >= len(questions):
            raise HTTPException(status_code=400, detail="Invalid question index.")

        question_data = questions[question_index]
        logger.info(f"Processing question {question_index}: {question_data}")

        audio_data = await audio_file.read()
        logger.info(f"Received audio file of length: {len(audio_data)} bytes")

        try:
            wav_data = evaluate_response.convert_to_wav(audio_data)
        except Exception as conv_err:
            logger.error(f"Error converting audio to WAV: {conv_err}")
            raise HTTPException(status_code=400, detail=f"Audio conversion failed: {conv_err}")

        # Step 1: Transcribe Audio
        try:
            response_text = evaluate_response.transcribe_audio(wav_data)
        except evaluate_response.TranscriptionError as e:
            logger.error(f"Transcription error: {e}")
            raise HTTPException(status_code=400, detail=f"Audio transcription failed: {e}")

        logger.info(f"Transcribed Response: {response_text}")

        # Step 2: Extract Relevant Keywords
        try:
            relevant_keywords, question_type, company_sector = evaluate_response.get_relevant_keywords(
                question_data, session["job_role"], session["company_name"], json.dumps(session["company_info"])
            )
            logger.info(f"Relevant Keywords: {relevant_keywords}, Question Type: {question_type}, Sector: {company_sector}")
        except Exception as e:
            logger.exception("Error getting relevant keywords.")
            raise HTTPException(status_code=500, detail=f"Error determining relevant keywords: {str(e)}")

        # Step 3: Score Response
        try:
            result = evaluate_response.score_response(
                response_text, question_data["question_text"], relevant_keywords, question_type, company_sector
            )
        except Exception as e:
            logger.exception("Error scoring response.")
            raise HTTPException(status_code=500, detail=f"Error scoring response: {str(e)}")

        logger.info(f"Evaluation Result: {result}")
        return JSONResponse(content={"feedback": result})

    except HTTPException as http_exception:
        logger.error(f"HTTP Exception: {http_exception.detail}")
        raise http_exception
    except Exception as e:
        logger.exception("Unexpected error in evaluate_response endpoint.")
        raise HTTPException(status_code=500, detail=f"An error occurred while evaluating the response.")

@app.post("/end_session/{session_id}")
def end_session(session_id: str):
    if session_id in session_data:
        del session_data[session_id]
        logger.info(f"Session {session_id} data removed from session_data.")
    return {"message": f"Session {session_id} ended successfully."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
