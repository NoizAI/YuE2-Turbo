#!/usr/bin/env bash
set -euo pipefail
root=/data/noiz-yue
image=sha256:d01d9819e7a45a70b3b134f0e3ba78892dac8d5fc821584b6dfc9a6472aa83f6
gpu=GPU-0ecee4af-a520-3813-78f6-19c6435de0ca
test -f "$root/models/download-complete.json"
if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo 'Assigned GPU already has a compute process; refusing to start.' >&2
  exit 1
fi
if ss -ltnH | awk '{print $4}' | grep -q ':8015$'; then
  echo 'Port 8015 is occupied; refusing to start.' >&2
  exit 1
fi
sudo -n docker network inspect noiz-yue-web-net >/dev/null 2>&1 ||
  sudo -n docker network create noiz-yue-web-net >/dev/null
sudo -n docker run -d --name noiz-yue-gpu5 --restart=unless-stopped --init \
  --network=noiz-yue-web-net \
  --gpus "device=$gpu" --cpus=4 --memory=48g --shm-size=2g --user=1000:1000 \
  --publish 127.0.0.1:8015:8000 --env-file "$root/service.env" \
  -e CUDA_VISIBLE_DEVICES="$gpu" -e HF_HOME=/opt/noiz-yue/hf-cache \
  -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -e TORCHINDUCTOR_CACHE_DIR=/opt/noiz-yue/torch-cache -e TRITON_CACHE_DIR=/opt/noiz-yue/triton-cache \
  -e PYTHONUNBUFFERED=1 -v "$root:/opt/noiz-yue" -w /opt/noiz-yue/source \
  --health-cmd='/opt/noiz-yue/venv/bin/python -c "import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:8000/health/ready\",timeout=4)"' \
  --health-interval=30s --health-timeout=5s --health-start-period=5m --health-retries=3 \
  --entrypoint /opt/noiz-yue/venv/bin/yue2-serve "$image"
