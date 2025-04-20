# Dockerfile

# 1. Base on a slim Python image
FROM python:3.12-slim

# 2. Install system deps (incl. ffmpeg)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      build-essential && \
    rm -rf /var/lib/apt/lists/*

# 3. Set workdir and copy in requirements
WORKDIR /app
COPY requirements.txt .

# 4. Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code
COPY . .

# 6. Expose port & launch
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
