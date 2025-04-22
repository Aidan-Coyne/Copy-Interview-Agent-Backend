# ─── STAGE 1: build & cache everything ────────────────────────────────────────
FROM python:3.12-slim AS builder

# 1. Install system deps for ffmpeg, git, build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      build-essential \
      git \
    && rm -rf /var/lib/apt/lists/*

# 2. Set cache dirs for spaCy & HF
ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HOME=/cache/huggingface \
    SPACY_CACHE=/cache/spacy

# 3. Copy and install Python requirements
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Pre‑download spaCy & HF models + verify ONNX runtime
RUN python - <<EOF
import spacy, onnxruntime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# spaCy English
spacy.cli.download("en_core_web_sm")

# ST embeddings (cache)
SentenceTransformer("all-MiniLM-L6-v2")

# NLI model (cache)
model_name = "roberta-large-mnli"
AutoTokenizer.from_pretrained(model_name)
AutoModelForSequenceClassification.from_pretrained(model_name)

# ONNX runtime sanity check
_ = onnxruntime.get_device()
EOF

# ─── STAGE 2: runtime image ──────────────────────────────────────────────────
FROM python:3.12-slim

# 1. Re‑create cache dirs & expose cache path
ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HOME=/cache/huggingface \
    SPACY_CACHE=/cache/spacy
RUN mkdir -p $TRANSFORMERS_CACHE $HF_HOME $SPACY_CACHE

# 2. Install only runtime OS deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy site‑packages, binaries & caches from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /cache /cache

# 4. Copy your application code
WORKDIR /app
COPY . .

# 5. Expose port & launch
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
