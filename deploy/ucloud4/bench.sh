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
sudo -n docker run -d --name noiz-yue-benchmark-gpu5 --gpus "device=$gpu" \
  --network=none --cpus=4 --memory=48g --shm-size=2g --user=1000:1000 \
  -e CUDA_VISIBLE_DEVICES="$gpu" -e HF_HOME=/opt/noiz-yue/hf-cache \
  -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -e TORCHINDUCTOR_CACHE_DIR=/opt/noiz-yue/torch-cache -e TRITON_CACHE_DIR=/opt/noiz-yue/triton-cache \
  -e PYTHONUNBUFFERED=1 -v "$root:/opt/noiz-yue" -w /opt/noiz-yue/source \
  --entrypoint /opt/noiz-yue/venv/bin/yue2-benchmark "$image" \
  --model /opt/noiz-yue/models/YuE2-3B --vae /opt/noiz-yue/models/YuE2-Vae \
  --requests /opt/noiz-yue/requests.jsonl --profiles reference resident --warmup 1 --repeats 2 \
  --memory-budget-gib 30 --output /opt/noiz-yue/results/benchmark-gpu5
