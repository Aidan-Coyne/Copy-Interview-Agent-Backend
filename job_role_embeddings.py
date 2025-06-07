from keyword_extraction import onnx_embedder
from job_role_library import job_role_library

def compute_role_embeddings():
    role_embeddings = {}
    for sector, roles in job_role_library.items():
        for role in roles:
            if not role.startswith("_"):
                embedding = onnx_embedder.embed([role.lower()])[0]
                role_embeddings[(sector, role)] = embedding
    return role_embeddings

ROLE_EMBEDDINGS = compute_role_embeddings()
print(f"✅ Precomputed {len(ROLE_EMBEDDINGS)} role embeddings.")
