FROM ghcr.io/brrradley/litelabs-worker:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV LITELABS_AUDIO_SEPARATOR_MODEL_DIR=/models/audio_separator

RUN apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config libsamplerate0-dev && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install audio-separator==0.44.2 onnxruntime-gpu==1.22.0
RUN mkdir -p /models/audio_separator

COPY handler.py /app/handler.py
COPY research_tools.py /app/research_tools.py
COPY research_bootstrap.py /app/research_bootstrap.py
COPY litelabs_research_slim_patch.py /app/litelabs_research_slim_patch.py
RUN python /app/litelabs_research_slim_patch.py

CMD ["python", "-u", "/app/research_bootstrap.py"]
