# Varuna FloodTwin — FastAPI backend for Hugging Face Spaces (Docker SDK).
# Serves the committed artifacts/patna bundle and runs the differentiable twin live on CPU.
# HF Spaces inject PORT and expect the app on 0.0.0.0:7860.
FROM python:3.11-slim

# rasterio wheels bundle libgdal but still dlopen system libexpat at runtime;
# build-essential is a compiler fallback for any sdist deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image small; everything else from PyPI wheels.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-deploy.txt

# App + the committed artifact bundle (dem, sinks, emulator.pt, figures, calibration, canal/storage plans).
COPY varuna/ ./varuna/
COPY api/ ./api/
COPY artifacts/ ./artifacts/

ENV VARUNA_WORK=artifacts/patna \
    VARUNA_PROJECT_ID=floodtwin \
    PORT=7860 \
    HF_HOME=/tmp/hf \
    MPLCONFIGDIR=/tmp/mpl
EXPOSE 7860

# Use shell form so $PORT (HF may override) is expanded.
CMD uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-7860}
