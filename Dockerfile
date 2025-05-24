# ─── STAGE 1: Build & cache everything ────────────────────────────────────────
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

# 3. Install Python packages (split for caching)
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir faster-whisper
RUN echo "✅ Installed Python requirements"

# 4. Preload models
RUN python - <<EOF
import spacy, onnxruntime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from faster_whisper import WhisperModel

print("📦 Downloading models...")
spacy.cli.download("en_core_web_sm")
SentenceTransformer("all-MiniLM-L6-v2")
AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
WhisperModel("tiny", download_root="/app/models", compute_type="int8")
_ = onnxruntime.get_device()
print("✅ Finished downloading models")
EOF

# 5. Clone llama.cpp and build static binary
RUN git clone https://github.com/ggerganov/llama.cpp.git /llama.cpp
WORKDIR /llama.cpp

# 6. Download GGUF model
RUN mkdir -p models && \
    wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -O models/tinyllama.gguf

# 7. Compile llama.cpp
RUN mkdir build && cd build && \
    cmake .. -DLLAMA_AVX2=ON -DLLAMA_AVX512=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF && \
    make -j"$(nproc)"

# 8. Copy compiled binary
RUN mkdir -p /llama/bin && \
    cp ./build/bin/llama-cli /llama/bin/llama && chmod +x /llama/bin/llama
RUN echo "✅ Built static llama binary"

# ─── STAGE 2: Runtime image ───────────────────────────────────────────────────
FROM python:3.12-slim

ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HOME=/cache/huggingface \
    SPACY_CACHE=/cache/spacy
RUN mkdir -p $TRANSFORMERS_CACHE $HF_HOME $SPACY_CACHE
RUN echo "✅ Created model cache directories"

# Install runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*
RUN echo "✅ Installed runtime system dependencies"

# Copy dependencies and models
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /cache /cache
COPY --from=builder /app/models /app/models
COPY --from=builder /llama.cpp/models/ /llama/models/
COPY --from=builder /llama/bin/llama /llama/bin/llama
RUN echo "✅ Copied static llama binary and model"

# App source
WORKDIR /app
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
