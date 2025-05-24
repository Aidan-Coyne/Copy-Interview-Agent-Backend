import os
import json
import hashlib
import firebase_admin
from firebase_admin import credentials, firestore

# --- 1. FIRESTORE INITIALIZATION USING ENV ---
def initialize_firestore_from_env():
    json_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not json_creds:
        raise RuntimeError("Missing GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable")

    cred_dict = json.loads(json_creds)
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)

    return firestore.client()

db = initialize_firestore_from_env()

# --- 2. UTILS ---
def get_question_hash(question: str) -> str:
    return hashlib.sha256(question.strip().encode()).hexdigest()

def cache_prompt_to_firestore(question: str, messages: list[dict], q_type: str = "unspecified"):
    doc_id = get_question_hash(question)
    db.collection("prompt_templates").document(doc_id).set({
        "question": question,
        "prompt": messages,
        "type": q_type
    }, merge=True)

def is_static_question(question: str) -> bool:
    dynamic_keywords = [
        "relevant_skills", "company_name", "company_sectors",
        "relevant_experience", "job_role", "random.choice"
    ]
    return not any(keyword in question for keyword in dynamic_keywords)

# --- 3. QUESTION DATA ---
# Paste your full `technical_questions`, `behavioral_questions`, and `situational_questions` lists here,
# as you've already written them. I'll just simulate the minimal static ones to illustrate:

technical_questions = [
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
    ("technical", "Describe your process for testing new features."),
    ("technical", "What tools do you use for version control and why?"),
    ("technical", "How do you manage technical debt in projects?"),
    ("technical", "Have you ever optimized a slow system? How did you do it?"),
    ("technical", "How do you ensure your work is secure?"),
    ("technical", "Describe a time when you had to quickly learn a new technology."),
    ("technical", "How do you collaborate with non-technical team members?")
]

# Behavioral and situational as-is
behavioral_questions = [
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
    ("situational", q) for q in [
        "How do you troubleshoot when a project isn’t progressing as planned?",
        "Imagine you must implement a new process with limited resources. How would you proceed?",
        "How would you handle a situation where a key team member underperforms?",
        "If you encountered conflicting priorities, how would you resolve them?",
        "What would you do if you found a major flaw right before a project deadline?",
        "Suppose you have to collaborate with a difficult colleague. What would you do?",
        "If given a project outside your expertise, how would you handle it?",
        "How would you manage project requirements that change mid-cycle?",
        "You’ve been asked to lead a team with clashing personalities. How would you handle it?",
        "If an unexpected client demand threatens your schedule, what would you do?",
        "What would be your first steps when starting a new project in an unfamiliar area?",
        "How would you handle a last-minute change to project scope?",
        "What would you do if a stakeholder kept changing their requests?",
        "You're assigned multiple urgent tasks. How would you prioritize them?",
        "How would you handle a situation where a deadline may be missed?",
        "What steps would you take if your team missed a key milestone?",
        "How would you respond if a client rejected your proposal?",
        "How would you begin a project with unclear objectives?",
        "What would you do if your team wasn’t collaborating effectively?",
        "You're asked to use unfamiliar software. How would you proceed?",
        "How would you deal with vague or conflicting instructions?",
        "You notice a recurring error in a colleague’s work. What do you do?",
        "You're asked to present to leadership on short notice. How do you prepare?",
        "What would you do if your team resisted an important change?",
        "If your project budget was suddenly cut, what would be your first move?",
        "A client provides critical feedback. How would you handle it?",
        "You're assigned to lead a failing project. What steps would you take?",
        "Two teammates strongly disagree on a solution. How do you respond?",
        "A key report is missing data right before submission. What do you do?",
        "Your manager is unavailable and a quick decision is needed. How do you act?",
        "How would you ensure a smooth handover when leaving a project?"
    ]
]

# --- 4. CACHE STATIC QUESTIONS TO FIRESTORE ---
all_questions = technical_questions + behavioral_questions + situational_questions

for q_type, question in all_questions:
    if not question or not is_static_question(question):
        print(f"❌ Skipped dynamic: {question}")
        continue

    # Build prompt messages (GPT format)
    messages = [
        {"role": "system", "content": "You are an AI career coach providing structured interview feedback."},
        {"role": "user", "content": f"""You are evaluating a candidate's interview answer.

Question:
\"{question}\"

Answer:
\"{{ANSWER_PLACEHOLDER}}\"

Provide feedback in two clear paragraphs:
1. What they did well (quote strong phrases).
2. What could be improved (clarify, expand, structure).
Keep it supportive, concise, and actionable."""}
    ]

    cache_prompt_to_firestore(question, messages, q_type)
    print(f"✅ Cached: {question}")
