FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV STEMFORGE_MODEL_DIR=/models/bs_roformer_sw
ENV LITELABS_AUDIO_SEPARATOR_MODEL_DIR=/models/audio_separator

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    build-essential \
    pkg-config \
    libsamplerate0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements.txt

COPY master_pack.py /app/master_pack.py
COPY handler.py /app/handler.py
COPY litelabs_live_patch.py /app/litelabs_live_patch.py
RUN python /app/litelabs_live_patch.py

CMD ["python", "-u", "/app/handler.py"]
