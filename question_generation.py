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
from keyword_extraction import extract_keywords
from io import BytesIO

logging.basicConfig(level=logging.INFO)
nlp = spacy.load("en_core_web_sm")

generic_skill_phrases = [
    "your problem-solving abilities", "your analytical skills", "your programming expertise",
    "your design experience", "your engineering knowledge", "your technical proficiencies",
    "your innovative thinking", "your collaborative skills", "your attention to detail",
    "your learning agility"
]

def get_text_to_speech_client():
    try:
        google_creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if not google_creds_json:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON not found.")

        creds_dict = json.loads(google_creds_json)
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        client = texttospeech.TextToSpeechClient(credentials=credentials)
        return client
    except Exception as e:
        logging.error(f"Failed to initialize TTS client: {e}")
        raise

def upload_audio_to_firebase(audio_bytes: bytes, filename: str, firebase_bucket) -> str:
    blob = firebase_bucket.blob(f"questions/{filename}")
    blob.upload_from_file(BytesIO(audio_bytes), content_type="audio/mpeg")
    blob.make_public()  # Optional, for direct frontend access
    return blob.public_url

def generate_questions(cv_text, company_info, job_role, company_name, selected_question_type="mixed", firebase_bucket=None):
    logging.info(f"Generating {selected_question_type} questions for {company_name} - {job_role}")

    relevant_skills = extract_relevant_skills_from_role(job_role)
    relevant_experience = extract_relevant_experience(cv_text)
    company_sectors = extract_sectors(company_info)

    if not relevant_skills:
        relevant_skills = random.sample(generic_skill_phrases, 2)
    relevant_experience = relevant_experience or "your field"

    cv_keywords = extract_keywords(cv_text, top_n=10)
    selected_cv_keywords = random.sample(cv_keywords, min(3, len(cv_keywords)))
    cv_insight_questions = []
    if len(selected_cv_keywords) >= 1:
        cv_insight_questions.append((
            "cv_insight",
            f"Your CV mentions '{selected_cv_keywords[0]}'. Can you elaborate on how this experience helped shape your skills?"
        ))
    if len(selected_cv_keywords) >= 2:
        cv_insight_questions.append((
            "cv_insight",
            f"How did your involvement with '{selected_cv_keywords[1]}' contribute to your professional development?"
        ))
    if len(selected_cv_keywords) >= 3:
        cv_insight_questions.append((
            "cv_insight",
            f"Could you share more context about your work related to '{selected_cv_keywords[2]}'?"
        ))

    technical_questions = [
        ("technical", f"Based on your CV, tell me about your experience with {relevant_skills[0]}."),
        ("technical", f"Can you describe a project where you demonstrated {relevant_skills[1]} skills?"),
        ("technical", f"Tell me about your most relevant experience in {relevant_experience}."),
        ("technical", f"How do your skills align with {company_name}'s work in {company_sectors}.") if company_sectors else None,
        ("technical", f"Can you describe a technical challenge relevant to the {company_sectors} sector and how you overcame it?") if company_sectors else None,
        ("technical", f"What are the most important technical trends shaping the {company_sectors} industry right now?") if company_sectors else None,
        ("technical", "Describe a situation where you had to overcome a challenging technical problem."),
        ("technical", "How do you stay updated with the latest advancements in your field?"),
        ("technical", "Explain a complex concept you recently learned and how it impacted your work."),
        ("technical", "What strategies do you use to ensure accuracy and quality?"),
        ("technical", "Can you provide an example of how you innovated to solve a persistent problem?")
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
        ("behavioral", f"What inspired you to pursue a career in {job_role}?"),
        ("behavioral", "Can you share an example of how you adapted to new tools or methods?"),
        ("behavioral", "Tell me about a project where your input was critical to its success."),
        ("behavioral", "Describe a time when you managed tight deadlines while maintaining high quality."),
        ("behavioral", "How do you typically handle feedback from supervisors or clients?"),
        ("behavioral", "Share an example of when you went above and beyond your job responsibilities."),
        ("behavioral", "How do you manage work-life balance when you're under pressure?")
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
        ("situational", "What would be your first steps when starting a new project in an unfamiliar area?")
    ]

    if selected_question_type.lower() == "technical":
        question_pool = technical_questions
    elif selected_question_type.lower() == "behavioral":
        question_pool = behavioral_questions
    elif selected_question_type.lower() == "situational":
        question_pool = situational_questions
    else:
        question_pool = technical_questions + behavioral_questions + situational_questions + cv_insight_questions

    question_pool = [q for q in question_pool if q and isinstance(q, tuple) and len(q) == 2]
    selected_questions = random.sample(question_pool, 8) if len(question_pool) > 8 else question_pool

    client = get_text_to_speech_client()
    voice_params = texttospeech.VoiceSelectionParams(
        language_code="en-GB",
        name="en-GB-Wavenet-B",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    question_data = []
    for i, (q_type, question) in enumerate(selected_questions):
        try:
            ssml_text = f"<speak><prosody rate='1' pitch='-10%'>{question}</prosody></speak>"
            synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
            response = client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config)

            if not response.audio_content:
                logging.error(f"No audio content returned for question {i + 1}: {question}")
                continue

            filename = f"{company_name.lower().replace(' ', '_')}_{job_role.lower().replace(' ', '_')}_q{i + 1}.mp3"
            firebase_url = upload_audio_to_firebase(response.audio_content, filename, firebase_bucket)
            logging.info(f"Uploaded question {i + 1} to Firebase: {firebase_url}")

            question_data.append({
                "question_text": question,
                "audio_file": firebase_url,
                "skills": relevant_skills,
                "experience": relevant_experience,
                "question_type": q_type
            })
        except Exception as e:
            logging.error(f"Error generating audio for question {i + 1}: {e}")

    return question_data

def extract_relevant_skills_from_role(job_role: str):
    job_role_lower = job_role.lower()
    for discipline, roles in job_role_library.items():
        if discipline.lower() in job_role_lower and "keywords" in roles:
            return random.sample(roles["keywords"], min(2, len(roles["keywords"])))
        for role_title, keywords in roles.items():
            if role_title.lower() in job_role_lower and isinstance(keywords, list):
                return random.sample(keywords, min(2, len(keywords)))
    return []

def extract_relevant_experience(cv_text: str):
    doc = nlp(cv_text)
    experience_terms = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "VERB"] and len(token.text) > 2]
    return ", ".join(experience_terms[:2]) if experience_terms else "your field"

def extract_sectors(company_info):
    matched_sectors = set()
    search_results = company_info.get("search_results", [])
    for item in search_results:
        snippet = item.get("snippet", "").lower()
        for sector, keywords in sector_library.items():
            for keyword in keywords:
                if fuzz.partial_ratio(keyword.lower(), snippet) > 75:
                    matched_sectors.add(sector)
    return ", ".join(matched_sectors) if matched_sectors else None
