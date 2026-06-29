FROM ghcr.io/brrradley/litelabs-worker:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV LITELABS_AUDIO_SEPARATOR_MODEL_DIR=/models/audio_separator

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libsamplerate0-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install audio-separator==0.44.2
RUN mkdir -p /models/audio_separator

CMD ["python", "-u", "/app/handler.py"]
