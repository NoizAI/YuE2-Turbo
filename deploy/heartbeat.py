import json
import os
import subprocess
import time
import urllib.error
import urllib.request


CONTROLLER = os.environ["YUE2_CONTROLLER"].rstrip("/")
REGISTER_TOKEN = os.environ["YUE2_REGISTER_TOKEN"]
WORKER_IP = os.environ.get("YUE2_WORKER_IP", "42.240.149.121")
WORKER_PORT = int(os.environ.get("YUE2_WORKER_PORT", "80"))
INSTANCE_ID = os.environ.get("YUE2_INSTANCE_ID", "ucloud-4-yue2-gpu5")
GPU_INDEX = os.environ.get("YUE2_GPU_INDEX", "5")
LOCAL_HEALTH = os.environ.get(
    "YUE2_LOCAL_HEALTH", "http://127.0.0.1:8015/health/ready"
)


def model_ready() -> bool:
    try:
        with urllib.request.urlopen(LOCAL_HEALTH, timeout=5) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def gpu_stats() -> dict:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={GPU_INDEX}",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            timeout=3,
            text=True,
        ).strip()
        util, used, total, temp, power = [
            float(value.strip()) for value in raw.split(",")
        ]
        return {
            "gpu_util": util / 100,
            "gpu_mem_used_mb": int(used),
            "gpu_mem_total_mb": int(total),
            "gpu_temp_c": int(temp),
            "gpu_power_w": power,
        }
    except Exception:
        return {}



def load_snapshot():
    with open(f"/data/noiz-yue/{'jobs' if GPU_INDEX == '5' else 'jobs-gpu' + GPU_INDEX}/load.json") as source:
        snapshot = json.load(source)
    if time.time() - snapshot["sampled_at"] > 10:
        raise RuntimeError("Worker load snapshot is stale")
    gpu_uuid = subprocess.check_output(["nvidia-smi", f"--id={GPU_INDEX}", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=3).strip()
    snapshot["physical_id"] = WORKER_IP + "/" + gpu_uuid
    return snapshot

def send_heartbeat() -> int:
    snapshot = load_snapshot()
    payload = {
        "instance_id": INSTANCE_ID,
        "ip": WORKER_IP,
        "port": WORKER_PORT,
        "token": REGISTER_TOKEN,
        "model": "yue2-3b",
        "provider": "ucloud",
        "gpu_class": "RTX 5090",
        "model_loaded": model_ready(),
        "max_concurrent": 4,
        "active_requests": snapshot["active_requests"],
        "queue_depth": snapshot["queue_depth"], "load_snapshot": snapshot,
        **gpu_stats(),
    }
    request = urllib.request.Request(
        f"{CONTROLLER}/_yue2-ucloud/heartbeat",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {REGISTER_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status


if __name__ == "__main__":
    while True:
        try:
            status = send_heartbeat()
            print(f"heartbeat_status={status}", flush=True)
        except Exception as exc:
            print(f"heartbeat_failed={type(exc).__name__}", flush=True)
        time.sleep(5)
