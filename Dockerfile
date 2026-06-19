FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV STEMFORGE_MODEL_DIR=/models/bs_roformer_sw

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --upgrade pip setuptools wheel --break-system-packages && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio --break-system-packages && \
    python3 -m pip install -r /app/requirements.txt --break-system-packages

COPY master_pack.py /app/master_pack.py
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
