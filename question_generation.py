import spacy
import random
import os
import json
import logging
from fuzzywuzzy import fuzz
from google.cloud import texttospeech
from google.oauth2 import service_account
from job_role_library import job_role_library
from sector_library import sector_library
from keyword_extraction import extract_keywords  # ONNX-backed extractor
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
nlp = spacy.load("en_core_web_sm")

def safe_insert(item: list[str] | None, index: int, fallback: str = "a relevant skill") -> str:
    if isinstance(item, list) and len(item) > index:
        value = item[index].strip()
        if value and 2 <= len(value) <= 50:
            return value
    return fallback

def safe_text(value: str | None, fallback: str = "your field") -> str:
    if value and isinstance(value, str) and 2 <= len(value.strip()) <= 50:
        return value.strip()
    return fallback

generic_skill_phrases = [
    "your problem-solving abilities", "your analytical skills", "your programming expertise",
    "your design experience", "your engineering knowledge", "your technical proficiencies",
    "your innovative thinking", "your collaborative skills", "your attention to detail",
    "your learning agility"
]

def get_text_to_speech_client():
    google_creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not google_creds_json:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS_JSON not found.")
    creds_dict = json.loads(google_creds_json)
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
    return texttospeech.TextToSpeechClient(credentials=credentials)

def upload_audio_to_firebase(audio_bytes: bytes, filename: str, firebase_bucket, session_id: str) -> str:
    blob = firebase_bucket.blob(f"sessions/{session_id}/audio_questions/{filename}")
    blob.upload_from_file(BytesIO(audio_bytes), content_type="audio/mpeg")
    blob.make_public()
    return blob.public_url

def generate_questions(
    cv_text: str,
    company_info: dict,
    job_role: str,
    company_name: str,
    selected_question_type: str = "mixed",
    firebase_bucket=None,
    session_id: str = None,
    cv_embeddings: any = None,
    cv_keywords: list = None
) -> list[dict]:
    logging.info(f"Generating {selected_question_type} questions for {company_name} - {job_role}")

    # 1) Extract keywords and context
    relevant_skills     = extract_relevant_skills_from_role(job_role)
    relevant_experience = extract_relevant_experience(cv_text)
    company_sectors     = extract_sectors(company_info)

    if not relevant_skills:
        relevant_skills = random.sample(generic_skill_phrases, 2)
    relevant_experience = relevant_experience or "your field"

    # ✅ Safe fallback logic
    def safe_insert(item: list[str] | None, index: int, fallback: str = "a relevant skill") -> str:
        if isinstance(item, list) and len(item) > index:
            value = item[index].strip()
            if value and 2 <= len(value) <= 50:
                return value
        return fallback

    def safe_text(value: str | None, fallback: str = "your field") -> str:
        if value and isinstance(value, str) and 2 <= len(value.strip()) <= 50:
            return value.strip()
        return fallback

    # ✅ Use safe values for dynamic question text
    skill_0    = safe_insert(relevant_skills, 0)
    skill_1    = safe_insert(relevant_skills, 1)
    experience = safe_text(relevant_experience, "a relevant technical area")
    sector     = safe_text(company_sectors, "the companies sector")
    company    = safe_text(company_name, "the company")


    # 2) Use CV keywords passed in (already extracted)
    selected_cv_keywords = random.sample(cv_keywords, min(2, len(cv_keywords))) if cv_keywords else []

    cv_insight_questions = []
    for idx, kw in enumerate(selected_cv_keywords):
        cv_insight_questions.append((
            "cv_insight",
            f"Your CV mentions '{kw}'. Can you elaborate on how this experience helped shape your skills?"
        ))

    # 3) Question pools
    technical_questions = [
        ("technical", f"Based on your CV, tell me about your experience with {relevant_skills[0]}.")
        ("technical", f"Can you describe a project where you demonstrated {relevant_skills[1]} skills?")
        ("technical", f"Tell me about your most relevant experience in {relevant_experience}."),
        ("technical", f"How do your skills align with {company_name}'s work in {company_sectors}.")
        ("technical", f"Can you describe a technical challenge relevant to the {company_sectors} sector and how you overcame it?")
        ("technical", "Describe a situation where you had to overcome a challenging technical problem."),
        ("technical", "How do you stay updated with the latest advancements in your field?"),
        ("technical", "Explain a complex concept you recently learned and how it impacted your work."),
        ("technical", "What strategies do you use to ensure accuracy and quality?"),
        ("technical", "Can you provide an example of how you innovated to solve a persistent problem?"),
        ("technical", "Tell me more about a technical achievement listed on your CV."),
        ("technical", "Can you elaborate on your role in the most recent project on your CV?"),
        ("technical", "What’s a technical challenge you described in your CV and how did you address it?"),
        ("technical", "Looking at your CV, which project best shows your ability to work independently?"),
        ("technical", "Which project on your CV best showcases your technical strengths?"),
        ("technical", "Can you walk me through a problem you solved that’s listed on your CV?"),
        ("technical", f"In your most recent project, how did you apply {relevant_skills[0]} and what was the result?") 
        ("technical", "Describe your process for testing new features."),
        ("technical", "What tools do you use for version control and why?"),
        ("technical", "How do you manage technical debt in projects?"),
        ("technical", "Have you ever optimized a slow system? How did you do it?"),
        ("technical", "How do you ensure your work is secure?"),
        ("technical", f"In your most recent project, how did you apply {relevant_skills[0]} and what was the result?")
        ("technical", f"What advice would you give someone starting to learn {relevant_skills[0]} based on your own experience?")
        ("technical", f"What challenges did you face while applying {relevant_skills[0]} in a real-world setting?") 
        ("technical", "Describe a time when you had to quickly learn a new technology."),
        ("technical", "How do you collaborate with non-technical team members?"),
        ("technical", f"How did you build your expertise in {relevant_skills[0]}?")
    ]

    behavioral_questions = [
        ("behavioral", f"What do you know about {company_name} and its values?"),
        ("behavioral", f"Why are you interested in the {job_role} role at {company_name}?"),
        ("behavioral", random.choice([
            "Give me an example of a time you had to learn something new quickly.",
            "Tell me about a time you faced a challenge and how you overcame it.",
            "Describe a situation where you worked in a team to achieve a goal.",
            "Tell me about a time you made a mistake and how you handled it."
        ])),
        ("behavioral", f"Can you give an example of when you used {random.choice(generic_skill_phrases)}?"),
        ("behavioral", f"How do you demonstrate {random.choice(generic_skill_phrases)} in your work?"),
        ("behavioral", f"What inspired you to pursue a career as a {job_role}?"),
        ("behavioral", "Can you share an example of how you adapted to new tools or methods?"),
        ("behavioral", "Tell me about a project where your input was critical to its success."),
        ("behavioral", "Describe a time when you managed tight deadlines while maintaining high quality."),
        ("behavioral", "How do you typically handle feedback from supervisors or clients?"),
        ("behavioral", "Share an example of when you went above and beyond your job responsibilities."),
        ("behavioral", "How do you manage work-life balance when you're under pressure?"),
        ("behavioral", "Describe a time when you had to handle a difficult stakeholder or client. How did you manage the relationship?"),
        ("behavioral", "Tell me about a situation where you had to make a decision with incomplete information."),
        ("behavioral", "Have you ever had to give constructive feedback to a colleague? How did you approach it?"),
        ("behavioral", "Can you share a time when you had to motivate a team or a colleague during a low point?"),
        ("behavioral", "Describe a project where you had to manage conflicting priorities. How did you handle it?"),
        ("behavioral", "Tell me about a time when you disagreed with a decision made by your manager. What did you do?"),
        ("behavioral", "Give an example of a goal you set and how you achieved it."),
        ("behavioral", "Describe a time when your initiative made a significant impact."),
        ("behavioral", "How have you handled working with someone whose style was very different from yours?"),
        ("behavioral", "Tell me about a time when you identified a problem before it became serious. What actions did you take?"),
        ("behavioral", "Tell me about a time you took initiative at work."),
        ("behavioral", "Describe a time you had to meet a tough deadline."),
        ("behavioral", "Have you ever led a team? What was the result?"),
        ("behavioral", "Tell me about a time you handled a conflict at work."),
        ("behavioral", "Describe a situation where you had to multitask."),
        ("behavioral", "How have you handled a task outside your job scope?"),
        ("behavioral", "Tell me about a time you improved a process."),
        ("behavioral", "Describe a time you had to explain something complex."),
        ("behavioral", "Have you ever made a difficult ethical decision?"),
        ("behavioral", "Tell me about a time you received unexpected feedback.")
    ]

    situational_questions = [
        ("situational", "How do you troubleshoot when a project isn’t progressing as planned?"),
        ("situational", "Imagine you must implement a new process with limited resources. How would you proceed?"),
        ("situational", "How would you handle a situation where a key team member underperforms?"),
        ("situational", "If you encountered conflicting priorities, how would you resolve them?"),
        ("situational", "What would you do if you found a major flaw right before a project deadline?"),
        ("situational", "Suppose you have to collaborate with a difficult colleague. What would you do?"),
        ("situational", "If given a project outside your expertise, how would you handle it?"),
        ("situational", "How would you manage project requirements that change mid-cycle?"),
        ("situational", "You’ve been asked to lead a team with clashing personalities. How would you handle it?"),
        ("situational", "If an unexpected client demand threatens your schedule, what would you do?"),
        ("situational", "What would be your first steps when starting a new project in an unfamiliar area?"),
        ("situational", "How would you handle a last-minute change to project scope?"),
        ("situational", "What would you do if a stakeholder kept changing their requests?"),
        ("situational", "You're assigned multiple urgent tasks. How would you prioritize them?"),
        ("situational", "How would you handle a situation where a deadline may be missed?"),
        ("situational", "What steps would you take if your team missed a key milestone?"),
        ("situational", "How would you respond if a client rejected your proposal?"),
        ("situational", "How would you begin a project with unclear objectives?"),
        ("situational", "What would you do if your team wasn’t collaborating effectively?"),
        ("situational", "You're asked to use unfamiliar software. How would you proceed?"),
        ("situational", "How would you deal with vague or conflicting instructions?"),
        ("situational", "You notice a recurring error in a colleague’s work. What do you do?"),
        ("situational", "You're asked to present to leadership on short notice. How do you prepare?"),
        ("situational", "What would you do if your team resisted an important change?"),
        ("situational", "If your project budget was suddenly cut, what would be your first move?"),
        ("situational", "A client provides critical feedback. How would you handle it?"),
        ("situational", "You're assigned to lead a failing project. What steps would you take?"),
        ("situational", "Two teammates strongly disagree on a solution. How do you respond?"),
        ("situational", "A key report is missing data right before submission. What do you do?"),
        ("situational", "Your manager is unavailable and a quick decision is needed. How do you act?"),
        ("situational", "How would you ensure a smooth handover when leaving a project?")
    ]

    # 4) Select and shuffle
    if selected_question_type.lower() == "technical":
        pool = technical_questions
    elif selected_question_type.lower() == "behavioral":
        pool = behavioral_questions
    elif selected_question_type.lower() == "situational":
        pool = situational_questions
    else:
        pool = technical_questions + behavioral_questions + situational_questions + cv_insight_questions

    pool = [q for q in pool if q]
    random.shuffle(pool)
    selected_questions = pool[:8]

    # 5) TTS config
    client = get_text_to_speech_client()
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-GB",
        name="en-GB-Wavenet-B",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )
    audio_conf = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    question_data: list[dict] = [None] * len(selected_questions)

    def _synth_and_upload(i: int, q_type: str, text: str):
        ssml = f"<speak><prosody rate='1' pitch='-10%'>{text}</prosody></speak>"
        resp = client.synthesize_speech(
            input=texttospeech.SynthesisInput(ssml=ssml),
            voice=voice,
            audio_config=audio_conf
        )
        if not resp.audio_content:
            logging.error(f"No audio for Q{i+1}")
            return

        fname = f"{company_name.lower().replace(' ', '_')}_{job_role.lower().replace(' ', '_')}_q{i+1}.mp3"
        url = upload_audio_to_firebase(resp.audio_content, fname, firebase_bucket, session_id)

        question_data[i] = {
            "question_text":    text,
            "audio_file":       url,
            "skills":           relevant_skills,
            "experience":       relevant_experience,
            "sector":           company_sectors,
            "question_type":    q_type,
            "role_keywords":    relevant_skills[:2],
            "sector_keywords":  company_sectors.split(", ")[:2] if company_sectors else [],
            "cv_keywords":      selected_cv_keywords[:2]
        }

    with ThreadPoolExecutor() as exe:
        futures = [
            exe.submit(_synth_and_upload, i, q_type, text)
            for i, (q_type, text) in enumerate(selected_questions)
        ]
        for _ in as_completed(futures):
            pass

    return question_data

def extract_relevant_skills_from_role(job_role: str):
    job_role_lower = job_role.lower()
    best_match = None
    best_score = 0
    for discipline, roles in job_role_library.items():
        if "_keywords" in roles and fuzz.partial_ratio(discipline.lower(), job_role_lower) > 80:
            return random.sample(roles["_keywords"], min(2, len(roles["_keywords"])))
        for role_title, keywords in roles.items():
            if isinstance(keywords, list):
                score = fuzz.partial_ratio(role_title.lower(), job_role_lower)
                if score > best_score:
                    best_match = keywords
                    best_score = score
    return random.sample(best_match, min(2, len(best_match))) if best_match else []

def extract_relevant_experience(cv_text: str):
    doc = nlp(cv_text)
    terms = [
        token.text.lower()
        for token in doc
        if token.pos_ in ["NOUN", "VERB"] and len(token.text) > 2
    ]
    return ", ".join(terms[:2]) if terms else "your field"

def extract_sectors(company_info: dict):
    matched_sectors = set()
    for item in company_info.get("search_results", []):
        snippet = item.get("snippet", "").lower()
        for sector, keywords in sector_library.items():
            for keyword in keywords:
                if fuzz.partial_ratio(keyword.lower(), snippet) > 75:
                    matched_sectors.add(sector)
    return ", ".join(matched_sectors) if matched_sectors else None
