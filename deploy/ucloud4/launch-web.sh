#!/usr/bin/env bash
set -euo pipefail

root=/data/noiz-yue
image=sha256:d01d9819e7a45a70b3b134f0e3ba78892dac8d5fc821584b6dfc9a6472aa83f6
public_port=${YUE2_PUBLIC_PORT:-80}

mkdir -p "$root/web-status"
sudo -n docker network inspect noiz-yue-web-net >/dev/null 2>&1 ||
  sudo -n docker network create noiz-yue-web-net >/dev/null
sudo -n docker rm -f noiz-yue-web >/dev/null 2>&1 || true
sudo -n docker run -d --name noiz-yue-web --restart=unless-stopped --init \
  --network=noiz-yue-web-net --user=1000:1000 \
  --publish "0.0.0.0:${public_port}:8016" \
  -e PYTHONPATH=/opt/noiz-yue/source/src \
  -e YUE2_UPSTREAM=http://noiz-yue-gpu5:8000 \
  -e YUE2_MAINTENANCE_FILE=/opt/noiz-yue/web-status/maintenance.txt \
  -e PYTHONUNBUFFERED=1 \
  -v "$root/source:/opt/noiz-yue/source:ro" \
  -v "$root/venv:/opt/noiz-yue/venv:ro" \
  -v "$root/web-status:/opt/noiz-yue/web-status:ro" \
  --health-cmd='/opt/noiz-yue/venv/bin/python -c "import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:8016/console/status\",timeout=4)"' \
  --health-interval=30s --health-timeout=5s --health-start-period=10s --health-retries=3 \
  --entrypoint /opt/noiz-yue/venv/bin/python "$image" -m yue2.web_gateway
