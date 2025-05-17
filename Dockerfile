# ─── STAGE 1: build & cache everything ────────────────────────────────────────
FROM python:3.12-slim AS builder

# 1. Install system deps for audio tools, CMake, OpenBLAS, git, and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      cmake \
      build-essential \
      libopenblas-dev \
      git \
      wget \
      unzip \
    && rm -rf /var/lib/apt/lists/*

# 2. Set cache dirs for HuggingFace and spaCy
ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HOME=/cache/huggingface \
    SPACY_CACHE=/cache/spacy

# 3. Copy and install Python dependencies
WORKDIR /build
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# 4. Install FasterWhisper and pre-cache the tiny model
RUN pip install --no-cache-dir faster-whisper

# 5. Pre-download spaCy & embedding models
RUN python - <<EOF
import spacy, onnxruntime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from faster_whisper import WhisperModel

# spaCy model
spacy.cli.download("en_core_web_sm")

# Sentence transformer (SBERT-style)
SentenceTransformer("all-MiniLM-L6-v2")

# Phi-2 tokenizer (cache only)
AutoTokenizer.from_pretrained("microsoft/phi-2")

# Whisper (tiny)
WhisperModel("tiny", download_root="/app/models", compute_type="int8")

# ONNX CPU sanity check
_ = onnxruntime.get_device()
EOF

# 6. Build llama.cpp with CMake for Phi-2
RUN git clone https://github.com/ggerganov/llama.cpp.git /llama.cpp && \
    cd /llama.cpp && mkdir build && cd build && \
    cmake .. -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS && \
    make -j$(nproc)

# ─── STAGE 2: runtime image ───────────────────────────────────────────────────
FROM python:3.12-slim

# Set environment variables
ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HOME=/cache/huggingface \
    SPACY_CACHE=/cache/spacy
RUN mkdir -p $TRANSFORMERS_CACHE $HF_HOME $SPACY_CACHE

# Install runtime OS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy Python site-packages, binaries, and cached models
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin               /usr/local/bin
COPY --from=builder /cache                       /cache
COPY --from=builder /app/models                  /app/models

# Copy llama.cpp compiled binaries
COPY --from=builder /llama.cpp/build/bin/        /llama/bin/
COPY --from=builder /llama.cpp/models/           /llama/models/

# Copy your application code
WORKDIR /app
COPY . .

# Expose port and run the app
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
