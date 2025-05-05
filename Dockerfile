# ─── STAGE 1: build & cache everything ────────────────────────────────────────
FROM python:3.12-slim AS builder

# 1. Install system deps for ffmpeg, git, build tools, plus wget & unzip for Vosk model download
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      build-essential \
      git \
      wget \
      unzip \
    && rm -rf /var/lib/apt/lists/*

# 2. Set cache dirs for spaCy & HF
ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HOME=/cache/huggingface \
    SPACY_CACHE=/cache/spacy

# 3. Copy and install Python requirements
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Pre-download spaCy, HF & ONNX models + verify ONNX runtime
RUN python - <<EOF
import spacy, onnxruntime
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)

# → spaCy English
spacy.cli.download("en_core_web_sm")

# → SBERT embeddings (cache)
SentenceTransformer("all-MiniLM-L6-v2")

# → NLI model (cache)
nli_model = "roberta-large-mnli"
AutoTokenizer.from_pretrained(nli_model)
AutoModelForSequenceClassification.from_pretrained(nli_model)

# → Small feedback LLM (cache)
feedback_model = "google/flan-t5-small"
AutoTokenizer.from_pretrained(feedback_model)
AutoModelForSeq2SeqLM.from_pretrained(feedback_model)

# → ONNX runtime sanity check
_ = onnxruntime.get_device()
EOF

# 5. Install Vosk and download a small English model
RUN pip install --no-cache-dir vosk && \
    mkdir -p /app/vosk-model && \
    wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -O /tmp/vosk.zip && \
    unzip /tmp/vosk.zip -d /app/vosk-model && \
    rm /tmp/vosk.zip

# ─── STAGE 2: runtime image ──────────────────────────────────────────────────
FROM python:3.12-slim

# 1. Re-create cache dirs & expose cache path
ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HOME=/cache/huggingface \
    SPACY_CACHE=/cache/spacy
RUN mkdir -p $TRANSFORMERS_CACHE $HF_HOME $SPACY_CACHE

# 2. Install only runtime OS deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy site-packages, binaries & caches from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /cache /cache

# 4. Copy the pre-downloaded Vosk model
COPY --from=builder /app/vosk-model /app/vosk-model

# 5. Copy your application code
WORKDIR /app
COPY . .

# 6. Expose port & launch
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
