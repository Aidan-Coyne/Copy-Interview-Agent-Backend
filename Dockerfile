# ─── STAGE 1: build & cache everything ────────────────────────────────────────
FROM python:3.12-slim AS builder

# 1. Install system deps
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

# 2. Set HuggingFace/spaCy cache dirs
ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HOME=/cache/huggingface \
    SPACY_CACHE=/cache/spacy

# 3. Install Python dependencies
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Install FasterWhisper and download model
RUN pip install --no-cache-dir faster-whisper

# 5. Download pretrained models
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

# 6. Build llama.cpp and rename binary to 'llama'
RUN git clone https://github.com/ggerganov/llama.cpp.git /llama.cpp && \
    wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf \
      -O /llama.cpp/models/phi-2.gguf && \
    cd /llama.cpp && mkdir build && cd build && \
    cmake .. -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_CURL=OFF && \
    make -j$(nproc) && \
    cp bin/main bin/llama  # 🔥 rename compiled binary to expected name

# ─── STAGE 2: runtime image ───────────────────────────────────────────────────
FROM python:3.12-slim

ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HOME=/cache/huggingface \
    SPACY_CACHE=/cache/spacy
RUN mkdir -p $TRANSFORMERS_CACHE $HF_HOME $SPACY_CACHE

# Install runtime deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy Python packages and models
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin               /usr/local/bin
COPY --from=builder /cache                       /cache
COPY --from=builder /app/models                  /app/models

# ✅ Copy compiled llama binary (renamed to llama) and model
COPY --from=builder /llama.cpp/build/bin/llama   /llama/bin/llama
COPY --from=builder /llama.cpp/models/           /llama/models/

# Copy app code
WORKDIR /app
COPY . .

# Run app
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
