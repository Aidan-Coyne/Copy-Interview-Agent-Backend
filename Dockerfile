# ─── STAGE 1: build & cache everything ────────────────────────────────────────
FROM python:3.12-slim AS builder

# 1. Install system deps for ffmpeg, git, build tools, plus wget/unzip
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

# 4. Install faster-whisper and pre-cache the tiny model
RUN pip install --no-cache-dir faster-whisper

# 5. Pre-download spaCy, HF, ONNX, Whisper & Vosk models
RUN python - <<EOF
import spacy, onnxruntime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from faster_whisper import WhisperModel

# → spaCy English
spacy.cli.download("en_core_web_sm")

# → SBERT embeddings (cache)
SentenceTransformer("all-MiniLM-L6-v2")

# → NLI model (cache)
AutoTokenizer.from_pretrained("roberta-large-mnli")
AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")

# → feedback LLM (FLAN-T5 base)
AutoTokenizer.from_pretrained("google/flan-t5-base")
AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# → ONNX sanity
_ = onnxruntime.get_device()

# → Whisper (tiny model)
WhisperModel("tiny", download_root="/app/models", compute_type="int8")
EOF

# 6. Download & unpack Vosk SMALL model (faster)
RUN mkdir -p /app/models && \
    wget -qO /app/models/vosk-model-small-en-us-0.15.zip \
      https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip && \
    unzip -q /app/models/vosk-model-small-en-us-0.15.zip -d /app/models && \
    rm /app/models/vosk-model-small-en-us-0.15.zip

# ─── STAGE 2: runtime image ──────────────────────────────────────────────────
FROM python:3.12-slim

ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HOME=/cache/huggingface \
    SPACY_CACHE=/cache/spacy
RUN mkdir -p $TRANSFORMERS_CACHE $HF_HOME $SPACY_CACHE

# install only runtime OS deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# copy Python site-packages, bin, cache and models
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin               /usr/local/bin
COPY --from=builder /cache                        /cache
COPY --from=builder /app/models                   /app/models

# copy your application code
WORKDIR /app
COPY . .

# expose & launch
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
