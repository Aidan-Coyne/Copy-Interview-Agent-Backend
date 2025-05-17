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

# 2. Set model cache directories
ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HOME=/cache/huggingface \
    SPACY_CACHE=/cache/spacy

# 3. Install Python packages
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir faster-whisper

# 4. Preload models for faster startup
RUN python - <<EOF
import spacy, onnxruntime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from faster_whisper import WhisperModel

spacy.cli.download("en_core_web_sm")
SentenceTransformer("all-MiniLM-L6-v2")
AutoTokenizer.from_pretrained("microsoft/phi-2")
WhisperModel("tiny", download_root="/app/models", compute_type="int8")
_ = onnxruntime.get_device()
EOF

# 5. Clone llama.cpp and build binary
RUN git clone https://github.com/ggerganov/llama.cpp.git /llama.cpp && \
    mkdir -p /llama.cpp/models && \
    wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf \
        -O /llama.cpp/models/phi-2.gguf && \
    cd /llama.cpp && mkdir build && cd build && \
    cmake .. -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_CURL=OFF && \
    make -j"$(nproc)" && \
    echo "🔍 Listing built binaries:" && \
    find . -type f -executable -exec ls -lh {} \; && \
    mkdir -p /llama/bin && \
    cp $(find . -type f -name "llama" -perm -111 | head -n 1) /llama/bin/llama || \
    (echo "❌ ERROR: 'llama' binary not found after build." && ls -R . && exit 1)

# ─── STAGE 2: minimal runtime image ───────────────────────────────────────────
FROM python:3.12-slim

# Set model cache environment
ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HOME=/cache/huggingface \
    SPACY_CACHE=/cache/spacy

RUN mkdir -p $TRANSFORMERS_CACHE $HF_HOME $SPACY_CACHE

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy Python environment and models
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /cache /cache
COPY --from=builder /app/models /app/models

# ✅ Copy the compiled llama binary and GGUF model
COPY --from=builder /llama.cpp/models /llama/models
COPY --from=builder /llama/bin/llama /llama/bin/llama

# Copy application code
WORKDIR /app
COPY . .

# Expose app port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
