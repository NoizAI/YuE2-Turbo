#!/usr/bin/env bash
set -euo pipefail
root=/data/noiz-yue
image=sha256:d01d9819e7a45a70b3b134f0e3ba78892dac8d5fc821584b6dfc9a6472aa83f6
gpu=GPU-0ecee4af-a520-3813-78f6-19c6435de0ca
if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo 'Assigned GPU already has a compute process; refusing to start.' >&2
  exit 1
fi
sudo -n docker run -d --name noiz-yue-gpu-tests --gpus "device=$gpu" \
  --network=none --cpus=4 --memory=24g --shm-size=2g --user=1000:1000 \
  -e CUDA_VISIBLE_DEVICES="$gpu" -e HF_HOME=/opt/noiz-yue/hf-cache \
  -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -e TORCHINDUCTOR_CACHE_DIR=/opt/noiz-yue/torch-cache -e TRITON_CACHE_DIR=/opt/noiz-yue/triton-cache \
  -e PYTHONUNBUFFERED=1 -v "$root:/opt/noiz-yue" -w /opt/noiz-yue/source \
  --entrypoint /bin/bash "$image" -lc \
  '/opt/noiz-yue/venv/bin/python -c "import torch; assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_properties(0)); print(torch.__version__, torch.version.cuda); print(torch.ones(8,device=\"cuda\").sum())" && /opt/noiz-yue/venv/bin/python -m pytest -q -rs'
