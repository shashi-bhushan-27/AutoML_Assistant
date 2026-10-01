FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# XGBoost and scikit-learn need the OpenMP runtime on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# CPU-only torch (pulled in by sentence-transformers for the retrieval embeddings), then the pinned stack
COPY requirements.txt .
RUN pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY . .
RUN chmod +x start_hf.sh

# Hugging Face Spaces routes traffic to 7860; the API listens on 8000
EXPOSE 7860
EXPOSE 8000

# GROQ_API_KEY must be provided at runtime (Space secret / --env-file), never baked into the image.
CMD ["./start_hf.sh"]
