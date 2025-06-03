from difflib import get_close_matches
from firebase_admin import firestore
import logging

def match_job_role(user_input: str, job_role_library: dict):
    user_input_lower = user_input.lower().strip()

    role_to_sector = {}
    all_roles = []

    # Flatten the job_role_library
    for sector, roles in job_role_library.items():
        for role in roles:
            if not role.startswith("_"):
                all_roles.append(role)
                role_to_sector[role] = sector

    # 1. Exact match
    for role in all_roles:
        if user_input_lower == role.lower():
            return role, role_to_sector[role]

    # 2. Substring match
    for role in all_roles:
        if role.lower() in user_input_lower:
            return role, role_to_sector[role]

    # 3. Fuzzy match
    best_match = get_close_matches(user_input_lower, [r.lower() for r in all_roles], n=1, cutoff=0.6)
    if best_match:
        for role in all_roles:
            if role.lower() == best_match[0]:
                return role, role_to_sector[role]

    # 4. Log unmatched role to Firestore
    try:
        db = firestore.client()
        unmatched_ref = db.collection("unmatched_roles").document(user_input_lower)
        unmatched_ref.set({
            "title": user_input,
            "last_seen": firestore.SERVER_TIMESTAMP,
            "count": firestore.Increment(1)
        }, merge=True)
        logging.info(f"📌 Logged unmatched job role: '{user_input}'")
    except Exception as e:
        logging.error(f"❌ Failed to log unmatched role '{user_input}': {e}")

    return None, None
