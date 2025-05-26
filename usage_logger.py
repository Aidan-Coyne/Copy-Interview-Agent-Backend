# usage_logger.py

import os
import requests
import logging

logger = logging.getLogger(__name__)

def log_usage_to_supabase(user_id: str, tokens_used: int, request_type: str):
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        logger.warning("Supabase credentials not found for logging.")
        return

    try:
        requests.post(
            f"{supabase_url}/rest/v1/usage_logs",
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            },
            json={
                "user_id": user_id,
                "tokens_used": tokens_used,
                "request_type": request_type
            }
        )
        logger.info(f"✅ Usage logged: {tokens_used} tokens for {request_type}")
    except Exception as e:
        logger.warning(f"❌ Failed to log usage: {e}")
