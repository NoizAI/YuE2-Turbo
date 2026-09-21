"""Compare real GPU runs. Never substitute synthetic timings for model measurements."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import time


PROFILES = {
    "reference": {"backend": "torch", "resident_models": False, "quantization": "none"},
    "resident": {"backend": "torch", "resident_models": True, "quantization": "none"},
    "vllm": {"backend": "vllm", "resident_models": True, "quantization": "none"},
    "fp8": {"backend": "torch", "resident_models": True, "quantization": "fp8"},
}


def summarize(records):
    """Report failures/truncation alongside timings; only complete songs enter latency stats."""
    complete = [r for r in records if r["status"] == "succeeded"]
    times = sorted(r["generation_seconds"] for r in complete)
    result = {"attempted": len(records), "succeeded": len(complete),
              "failed": sum(r["status"] == "failed" for r in records),
              "truncated": sum(r["status"] == "truncated" for r in records)}
    if times:
        result.update(p50_generation_seconds=statistics.median(times),
                      p95_generation_seconds=times[math.ceil(.95 * len(times)) - 1],
                      mean_generation_seconds=statistics.mean(times),
                      median_rtf=statistics.median(r["rtf"] for r in complete),
                      timing_population="succeeded only; warmup/save excluded",
                      p95_method="nearest rank")
    return result


def environment(device="cuda"):
    import torch
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(), "device": device}
    if device.startswith("cuda") and torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(device)
        result.update(gpu=prop.name, gpu_memory_gib=prop.total_memory / 2**30,
                      compute_capability=list(torch.cuda.get_device_capability(device)),
                      bf16_supported=torch.cuda.is_bf16_supported())
    return result


def load_requests(path):
    from .protocol import SongRequest
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        requests = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        requests = data if isinstance(data, list) else [data]
    if not requests:
        raise ValueError("Request set is empty")
    return [SongRequest(**request).to_dict() for request in requests]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Print environment only; no model downloads")
    parser.add_argument("--requests", type=Path, default=Path("examples/song.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=["reference", "resident"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1, help="Full warmup runs per profile, excluded from summary")
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--vae", default="m-a-p/YuE2-Vae")
    parser.add_argument("--revision")
    parser.add_argument("--vae-revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-budget-gib", type=float, default=30)
    parser.add_argument("--ode-steps", type=int, default=32)
    args = parser.parse_args(argv)
    env = environment(args.device)
    if args.check:
        print(json.dumps(env, indent=2))
        return 0 if env.get("cuda_available") and env.get("bf16_supported") else 1
    if args.repeats < 1 or args.warmup < 0 or args.ode_steps < 1:
        parser.error("repeats and ode-steps must be positive; warmup must be nonnegative")
    if len(set(args.profiles)) != len(args.profiles):
        parser.error("profiles must not repeat")
    if not args.device.startswith("cuda") or not env.get("cuda_available"):
        parser.error("A CUDA GPU is required for this performance benchmark")
    from .pipeline import YuE2Pipeline
    from .protocol import GenerationConfig
    from .sampling import synchronize
    import torch

    requests = load_requests(args.requests)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "environment": env,
              "note": "Sequential single-candidate GPU runs; no queue/upload. PyTorch peaks exclude other processes and non-PyTorch allocations.",
              "requests": requests, "profiles": {}}
    failures = False

    def save_report():
        temporary = args.output / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(args.output / "report.json")

    for name in args.profiles:
        profile = report["profiles"][name] = {"options": PROFILES[name], "records": []}
        pipe = None
        try:
            started = time.perf_counter()
            pipe = YuE2Pipeline.from_pretrained(args.model, vae=args.vae, revision=args.revision,
                       vae_revision=args.vae_revision, device=args.device, **PROFILES[name],
                       memory_budget_gib=args.memory_budget_gib,
                       generation_config=GenerationConfig(ode_steps=args.ode_steps), progress=False)
            pipe.preload()
            profile["preload_seconds"] = time.perf_counter() - started
            profile["weights"] = pipe.weights
            started = time.perf_counter()
            for _ in range(args.warmup):
                pipe(**requests[0])
            synchronize(args.device)
            profile["warmup_seconds"] = time.perf_counter() - started
            for repeat in range(args.repeats):
                for index, request in enumerate(requests):
                    record = {"request_index": index, "repeat": repeat, "seed": request["seed"]}
                    try:
                        torch.cuda.reset_peak_memory_stats(args.device)
                        synchronize(args.device)
                        started = time.perf_counter()
                        song = pipe(**request)
                        synchronize(args.device)
                        seconds = time.perf_counter() - started
                        duration = len(song.audio) / song.sample_rate
                        record.update(status="truncated" if any(song.truncated.values()) else "succeeded",
                                      generation_seconds=seconds, audio_seconds=duration, rtf=seconds / duration,
                                      peak_allocated_gib=torch.cuda.max_memory_allocated(args.device) / 2**30,
                                      peak_reserved_gib=torch.cuda.max_memory_reserved(args.device) / 2**30,
                                      timing=song.timing, configuration=song.config, truncated=song.truncated)
                        if record["status"] == "truncated":
                            failures = True
                        started = time.perf_counter()
                        directory = args.output / name / f"{repeat:03d}-{index:03d}"
                        song.save_artifacts(directory)
                        record.update(save_seconds=time.perf_counter() - started,
                                      artifacts=str(directory.relative_to(args.output)))
                        del song
                    except Exception as error:
                        record.update(status="failed", error=f"{type(error).__name__}: {error}")
                        failures = True
                        # Avoid reporting later samples from a potentially damaged runtime.
                        profile["records"].append(record)
                        raise
                    profile["records"].append(record)
                    profile["summary"] = summarize(profile["records"])
                    save_report()
                    print(f"{name} repeat={repeat} request={index}: {record['status']} {seconds:.2f}s", flush=True)
        except Exception as error:
            failures = True
            profile["error"] = f"{type(error).__name__}: {error}"
        finally:
            if pipe is not None:
                pipe.close()
            profile["summary"] = summarize(profile["records"])
            save_report()
    print(f"Report: {args.output / 'report.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
