# Linux/amd64 + CUDA 12.8 for RTX 5090. Weights use a mounted cache.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/data/huggingface \
    YUE2_DATA_DIR=/data/jobs \
    YUE2_HOST=0.0.0.0 \
    YUE2_PORT=8000
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv ca-certificates libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
WORKDIR /app
COPY . /app
RUN uv pip install --python /opt/venv/bin/python \
        torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install --python /opt/venv/bin/python '.[server]'
# Named volumes inherit ownership on first use; bind mounts need matching permissions.
RUN useradd --uid 10001 --create-home yue2 && mkdir -p /data/huggingface /data/jobs \
    && chown -R yue2:yue2 /data
USER yue2
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4)"
CMD ["yue2-serve"]
