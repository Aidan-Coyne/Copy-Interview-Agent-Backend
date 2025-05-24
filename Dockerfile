# ─── STAGE 1: build & cache everything ────────────────────────────────────────
FROM python:3.12-slim AS builder

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    cmake \
    build-essential \
    libopenblas-dev \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*
RUN echo "✅ Installed system dependencies"

# 2. Set model cache directories
ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HOME=/cache/huggingface \
    SPACY_CACHE=/cache/spacy
RUN echo "✅ Set model cache environment variables"

# 3. Install Python packages
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN echo "✅ Installed Python requirements"

# 4. Preload models (no TinyLlama)
RUN python - <<EOF
import spacy, onnxruntime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from faster_whisper import WhisperModel

print("📦 Downloading models...")
spacy.cli.download("en_core_web_sm")
SentenceTransformer("all-MiniLM-L6-v2")
AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")
WhisperModel("tiny", download_root="/app/models", compute_type="int8")
_ = onnxruntime.get_device()
print("✅ Finished downloading models")
EOF

# ─── STAGE 2: minimal runtime image ───────────────────────────────────────────
FROM python:3.12-slim

# Set up environment caches
ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HOME=/cache/huggingface \
    SPACY_CACHE=/cache/spacy
RUN mkdir -p $TRANSFORMERS_CACHE $HF_HOME $SPACY_CACHE
RUN echo "✅ Created model cache directories"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*
RUN echo "✅ Installed runtime system dependencies"

# Copy from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /cache /cache
COPY --from=builder /app/models /app/models

# Copy app source code
WORKDIR /app
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
