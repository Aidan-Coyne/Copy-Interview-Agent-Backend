from job_role_embeddings import ROLE_EMBEDDINGS
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from firebase_admin import firestore
import logging
from keyword_extraction import onnx_embedder  # still used for input

def match_job_role(user_input: str, job_role_library: dict, threshold=0.55):
    user_input_lower = user_input.lower().strip()
    role_to_sector = {role: sector for (sector, role) in ROLE_EMBEDDINGS.keys()}
    all_roles = [role for (_, role) in ROLE_EMBEDDINGS.keys()]

    # 1. Exact match
    for role in all_roles:
        if user_input_lower == role.lower():
            return role, role_to_sector[role]

    # 2. Substring match
    for role in all_roles:
        if role.lower() in user_input_lower or user_input_lower in role.lower():
            return role, role_to_sector[role]

    # 3. Fuzzy match
    best_match = get_close_matches(user_input_lower, [r.lower() for r in all_roles], n=1, cutoff=0.65)
    if best_match:
        for role in all_roles:
            if role.lower() == best_match[0]:
                return role, role_to_sector[role]

    # 4. Semantic match using precomputed role embeddings
    try:
        input_vec = onnx_embedder.embed([user_input_lower])[0]
        best_score = 0
        best_role = None

        for (sector, role), role_vec in ROLE_EMBEDDINGS.items():
            score = cosine_similarity([input_vec], [role_vec])[0][0]
            if score > best_score:
                best_score = score
                best_role = role

        if best_score >= threshold:
            return best_role, role_to_sector[best_role]
    except Exception as e:
        logging.warning(f"⚠️ Semantic matching failed: {e}")

    # 5. Log unmatched role
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
