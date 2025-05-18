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
RUN pip install --no-cache-dir faster-whisper
RUN echo "✅ Installed Python requirements"

# 4. Preload models for faster startup
RUN python - <<EOF
import spacy, onnxruntime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from faster_whisper import WhisperModel

print("📦 Downloading models...")
spacy.cli.download("en_core_web_sm")
SentenceTransformer("all-MiniLM-L6-v2")
AutoTokenizer.from_pretrained("microsoft/phi-2")
WhisperModel("tiny", download_root="/app/models", compute_type="int8")
_ = onnxruntime.get_device()
print("✅ Finished downloading models")
EOF

# 5. Clone llama.cpp and build the CLI
RUN git clone https://github.com/ggerganov/llama.cpp.git /llama.cpp
WORKDIR /llama.cpp

RUN mkdir -p models && \
    wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf \
    -O models/phi-2.gguf

RUN mkdir build && cd build && \
    cmake .. -DLLAMA_AVX2=ON -DLLAMA_AVX512=OFF -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_CURL=OFF && \
    make -j"$(nproc)"

RUN echo "🔍 Contents of build/bin:" && ls -lh ./build/bin

# Copy CLI binary and shared library
RUN mkdir -p /llama/bin && \
    cp ./build/bin/llama-cli /llama/bin/llama && \
    cp ./build/bin/libllama.so /llama/bin/

RUN echo "✅ Built llama CLI and copied shared libs"

# ─── STAGE 2: minimal runtime image ───────────────────────────────────────────
FROM python:3.12-slim

ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HOME=/cache/huggingface \
    SPACY_CACHE=/cache/spacy \
    LD_LIBRARY_PATH=/llama/bin
RUN mkdir -p $TRANSFORMERS_CACHE $HF_HOME $SPACY_CACHE
RUN echo "✅ Created model cache directories"

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*
RUN echo "✅ Installed runtime system dependencies"

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /cache /cache
COPY --from=builder /app/models /app/models

COPY --from=builder /llama.cpp/models/ /llama/models/
COPY --from=builder /llama/bin/ /llama/bin/
RUN echo "✅ Copied llama binary and shared libraries"

WORKDIR /app
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
