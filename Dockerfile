FROM ghcr.io/brrradley/litelabs-worker:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV LITELABS_AUDIO_SEPARATOR_MODEL_DIR=/models/audio_separator
ENV LITELABS_RESEARCH_BUILD=benchmark-suite-2

RUN apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config libsamplerate0-dev && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install audio-separator==0.44.2 onnxruntime-gpu==1.22.0
RUN mkdir -p /models/audio_separator

COPY handler.py /app/handler.py
COPY research_tools.py /app/research_tools.py
COPY benchmark_suite.py /app/benchmark_suite.py
COPY research_bootstrap.py /app/research_bootstrap.py
COPY litelabs_audio_separator_diagnostics_patch.py /app/litelabs_audio_separator_diagnostics_patch.py
RUN python /app/litelabs_audio_separator_diagnostics_patch.py
RUN test -f /app/benchmark_suite.py && python -c "import sys; sys.path.insert(0, '/app'); import benchmark_suite; print('benchmark_suite import ok')"

CMD ["python", "-u", "/app/research_bootstrap.py"]
