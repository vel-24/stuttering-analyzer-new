# Hugging Face Space (Docker SDK) for the Python inference service.
FROM python:3.11-slim

# ffmpeg: decode browser recordings (.webm). libsndfile1: used by librosa/soundfile.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
# The extra index offers the CPU-only PyTorch build (small, fits the free CPU tier);
# pip prefers it over the large default build, and falls back to PyPI if it is missing.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Bake the base Whisper weights into the image so the first request doesn't have to download them.
RUN python -c "from transformers import WhisperModel, WhisperFeatureExtractor; \
WhisperModel.from_pretrained('openai/whisper-base'); WhisperFeatureExtractor.from_pretrained('openai/whisper-base')"

COPY --chown=user . .
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
