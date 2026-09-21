# Experimental batch=2 inference

`YuE2Pipeline.generate_batch([request_a, request_b])` shares each autoregressive
model forward across two independent songs. The symbolic and semantic stages
use separate token histories, random generators, end conditions, KV-cache rows
and position counters. This is real shared-model batching, not Python threads.
NAR synthesis and VAE decoding still run sequentially.

```python
from yue2 import YuE2Pipeline

pipe = YuE2Pipeline.from_pretrained(
    "/path/to/YuE2-3B", vae="/path/to/YuE2-Vae", device="cuda",
    resident_models=True, memory_budget_gib=30, local_files_only=True,
)
try:
    songs = pipe.generate_batch([
        {"style": "piano pop", "lyrics": "[Verse]\nHello morning", "seed": 42},
        {"style": "acoustic pop", "lyrics": "[Verse]\nFollow the sunlight", "seed": 81},
    ])
    for i, song in enumerate(songs):
        song.save_artifacts(f"results/song-{i}")
finally:
    pipe.close()
```

Limits:

- Exactly two requests; torch CUDA, resident BF16, no quantization or AR offload.
- `cot=full` or `melody`, effective CFG=1. External ABC is accepted; when only one
  request needs planning, that planning uses the existing single-song path.
- Full results return together. A completed row stays in the batch with ignored
  dummy tokens until the other row finishes; there is no continuous admission.
  Pairing similar lengths generally wastes less work.
- Do not call the same pipeline concurrently. Cancellation aborts the whole pair;
  `on_token(row, phase, token)` identifies each row, `on_stage(stage)` tracks phases.
- Batch shape can change BF16 rounding and thus sampled songs despite equal seeds.
  Tests establish row isolation, not perceptual equivalence of all outputs.
- Timing fields `abc.seconds` and `semantic.seconds` are shared barrier durations,
  not additive per-song durations. `row_finished_seconds` describes each AR row;
  `batch_item_ready_seconds` is internal readiness and `batch_return_seconds` is
  actual pair completion. Whole-call timing excludes saving artifacts.
- The FastAPI service admission/execution queue remains unchanged. This opt-in
  prototype does not automatically gather HTTP jobs into batches. Production
  integration still needs queue deadlines, per-job cancellation and failure policy.

## Reproduce the paired experiment

`tools/benchmark_batch2.py` requires a dedicated GPU and a root directory with:
`models/YuE2-3B`, `models/YuE2-Vae`, `requests.jsonl` containing the short then long
SongRequest, and `results/benchmark-gpu5/reference/000-000` and `000-001`, each with
`prefix.npy` and at least 512 `semantic.npy` tokens from prior single-song results.
Inputs and those reference traces are preserved with the experiment evidence.

```bash
python tools/benchmark_batch2.py --root /opt/noiz-yue --output /opt/noiz-yue/experiments/batch2-new
```

The command saves three warmup songs, 16 measured songs, input requests, timings,
model identities and a report. Short pairs are tested twice with reversed order;
long and mixed pairs once each. Equal-work traces run three repetitions per mode
with 1,024 total forced forward tokens, excluding sampling and prefill. This
separates compute acceleration from changed generated lengths. Six logit
checkpoints compare numeric differences. It is a small benchmark, not a load SLA.

GPU utilization means the fraction of sampling time with a kernel active; it is
not SM occupancy. Memory utilization means access-busy time, not bandwidth saturation.
CPU 100% means one core. Compare successful, non-truncated output durations and
fixed-work timings together before selecting a production batching strategy.

## 2026-09-15: ucloud-4 GPU 5 measurements

RTX 5090, torch 2.10.0+cu128, resident BF16, 32 ODE steps, 4 CPU quota.
16 measured songs plus 3 warmups; all finished without truncation or OOM.
GPU suite: 219 passed, 2 skipped, 30 subtests. CPU suite: 209 passed,
12 skipped, 30 subtests. Optional vLLM and original release comparisons skipped.

- short-short: mean pair wall time 33.24 → 24.52 s; RPM 3.610 → 4.894. Output seconds [69.95866666666667, 69.95866666666667] → [68.79866666666666, 68.79866666666666]. Audio-duration-normalized throughput gain 33.3%.
- long-long: mean pair wall time 123.45 → 69.37 s; RPM 0.972 → 1.730. Output seconds [242.23866666666666, 242.23866666666666] → [181.39866666666666, 181.39866666666666]. Audio-duration-normalized throughput gain 33.3%.
- mixed: mean pair wall time 78.40 → 59.64 s; RPM 1.531 → 2.012. Output seconds [69.95866666666667, 242.23866666666666] → [66.87866666666666, 181.39866666666666]. Audio-duration-normalized throughput gain 4.5%.

Short pairs were repeated twice with reversed ordering; long/mixed pairs once.
Equal-work forward-only throughput improved 56.5% (1,024 forced tokens,
5.024 → 3.211 s, three repeats each). This excludes prefill and sampling.
Batch generation changes outputs, notably long songs from 242.24 to 181.40 s;
the raw long-song RPM increase is **not** an equal-output speedup. Even duration
normalization is only a workload proxy, not a quality or lyric-coverage metric.

Batch NVML sampled memory peak 17.59 GiB, torch allocated peak 10.02 GiB,
CPU about 1 core, no cgroup CPU throttling. This single-process experiment shares
the CUDA allocator across modes, so reserved/observed memory includes history;
these values are not independent cold-start memory requirements.

Recommendation from these samples: batch similar-length work when throughput
matters; mixed lengths have little normalized gain and delay the short song.
Further HTTP batching must account for waiting latency and output-quality checks.
The original service was restored in serial mode; the seven other workers kept
the same PID and start time. Raw inputs, outputs, metrics and tested source archive
are in `/Users/jishenwei/workFile/2026/0914yue2/batch2-01`.
See [machine-readable analysis](benchmarks/ucloud4-5090-batch2.json).
