# upload_onnx_to_firebase.py
import os
import tempfile
import firebase_admin
from firebase_admin import credentials, storage

# 🔑 Uses your existing GOOGLE_APPLICATION_CREDENTIALS_JSON env var
google_creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if not google_creds_json:
    raise RuntimeError("Set GOOGLE_APPLICATION_CREDENTIALS_JSON before running")

# write the JSON to a temp file so firebase_admin can use it
with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as fp:
    fp.write(google_creds_json.encode())
    fp_path = fp.name

# init Firebase
cred = credentials.Certificate(fp_path)
firebase_admin.initialize_app(cred, {
    "storageBucket": "ai-interview-agent-e2f7b.firebasestorage.app"
})
bucket = storage.bucket()

# upload our ONNX
local_path = "keyword_extraction/models/paraphrase-MiniLM-L3-v2.onnx"
remote_path = "models/paraphrase-MiniLM-L3-v2.onnx"
blob = bucket.blob(remote_path)
blob.upload_from_filename(local_path)
blob.make_public()

print("✅ Uploaded ONNX to Firebase at:")
print("   ", blob.public_url)
