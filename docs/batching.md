# Variable-size shared-model batching

`YuE2Pipeline.generate_batch(requests)` now accepts 2–64 independent requests,
subject to conservative CUDA memory admission. The implementation batches AR
forwards; NAR synthesis and VAE decoding remain sequential. The Python maximum
64 is an API bound, **not** an advertised safe batch size on a 5090.

The prior [batch=2 experiment](batch2.md) remains a historical result. Its limits
of exactly two requests describe that earlier revision. Existing FastAPI workers
still execute jobs serially; this API does not change HTTP queue admission.

## Memory admission

Before allocating any batch KV cache, `batch_memory_estimate` computes
`2 × layers × batch × (longest prefix + token budget) × KV heads × head dim × dtype bytes`.
It budgets another 20% of KV bytes and 2 GiB scratch, preserves 4 GiB physical
headroom, and stays 1 GiB below PyTorch's process allocator limit. Reclaimable
PyTorch cache counts as available memory. `BatchAdmissionError` rejects a batch
before KV allocation if this estimate exceeds available memory.

This is a conservative preflight check, not a guarantee about every CUDA driver,
allocator fragment or concurrent external GPU process. Input lengths, maximum
generation budgets, dtype, other resident models and device memory all affect
admission. Do not discover capacity by repeatedly provoking OOM.

The sweep uses the longest test input plus the **maximum** configured ABC length
when planning semantic-cache admission. It checks actual prefixes again in each
AR stage. NAR/VAE execute one song at a time after the AR cache is released.

## Semantics

- torch CUDA, resident BF16, no quantization or AR offload; cot=full/melody, CFG=1.
- Independent tokens, cache rows/positions, random generators, penalties and EOS.
- All results return together. Finished rows keep dummy tokens until the others
  finish. No continuous admission; arbitrary concurrent calls are unsupported.
- Cancellation aborts the entire batch. `on_token(row,phase,token)` distinguishes
  rows. An exception fails the call; it does not silently return a partial batch.
- Batch shape can change floating-point rounding and sampled songs. Result config
  includes the group identity and row index as part of the result identity.
- Shared AR phase durations are not additive per-song compute durations.

## Reproduce the sweep

```bash
python tools/benchmark_batch_sweep.py --root /opt/noiz-yue --output /opt/noiz-yue/experiments/batch-sweep-new
```

The root must contain the two models, two ordered requests in `requests.jsonl`
(short then long), and `results/benchmark-gpu5/reference/000-001/prefix.npy` and
`semantic.npy` (at least 256 tokens). Use a dedicated GPU with an explicit device
mask and preserve input/output evidence. Do not run against a GPU serving traffic.

The tool calculates an input/budget-specific maximum, probes **every integer**
from 1 through that maximum with fixed 256-token traces (three repetitions,
forward-only and forward-with-sampling costs), then generates complete short and
long workloads for each size. Each batch uses identical inputs/seeds across rows
so batch size is the controlled factor. Candidates leading either raw RPM or
output-duration-normalized throughput are repeated in descending size order.

Forward-with-sampling probes execute sampling but force the known teacher tokens
into the next step, maintaining equal contexts. They are not generated songs and
must not be reported as completed-task RPM. Real-song RPM counts only outputs
without truncation, excludes saving/HTTP/queue delay, and must be interpreted
alongside output durations. The same input may produce different music or omit
lyrics; successful termination is not a perceptual-quality assessment.

## 2026-09-15 RTX 5090 result

The conservative admission limit was **batch=10**, using the longest input plus
maximum ABC budget (4,505 prefix tokens total), 9,000 semantic tokens, BF16,
resident weights and a 30 GiB pipeline budget. Batch=11 failed the estimate and
was **not allocated or executed**. This is a safe testing ceiling for these
inputs/settings, not an experimentally determined absolute hardware maximum.

Every integer 1–10 was tested with complete short and long songs. Leaders 9 and
10 were repeated in reverse order. 148 measured songs plus one warmup completed;
no truncation or OOM. Final GPU regression: 232 passed, 2 skipped, 30 subtests;
local regression: 218 passed, 16 skipped, 30 subtests.

Batch=10 won raw RPM and duration-normalized throughput in both workloads:

- Short: **8.564 RPM**, ten results return in **70.06 s**; each output 74.92 s.
  Batch=9: 8.128 RPM, nine results in 66.43 s; each output 77.60 s.
- Long: **3.061 RPM**, ten results return in **196.04 s**; each output 184.32 s.
  Batch=9: 2.987 RPM, nine results in 180.75 s; each output 184.32 s.
- Equal short/long task counts grouped separately: batch=10 about **4.510 RPM**,
  excluding batch-formation waiting. This is a stated workload assumption.
- Peak sampled NVML memory: **26.63 GiB**, leaving about 5.2 GiB on this device;
  CPU around one core, no CPU quota throttling.

Batch=10 improves on 9 by roughly 5.4% short raw RPM and 2.4% long raw RPM.
Short output duration also changed, so its duration-normalized gain is only
about 1.7%. For throughput priority, 10 is the best measured admitted choice.
It increases per-batch latency; it is not the lowest-latency serving choice.
Output content and lyric coverage are not proven equivalent across batch sizes.

The original HTTP worker was restored in serial mode. The seven other workers
kept their PID/start time. Automatic HTTP grouping remains separate work.
See [all measurements](benchmarks/ucloud4-5090-batch-sweep.json). Full evidence is
under `/Users/jishenwei/workFile/2026/0914yue2/batch-sweep-01`.
