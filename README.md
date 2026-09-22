<p align="center">
  <img src="assets/yue2-turbo-logo.png" alt="YuE2 Turbo" width="460" />
</p>

<p align="center">
  <b>English</b> | <a href="README_zh.md">中文</a>
</p>

<h1 align="center">YuE2-Turbo: Fast, Concurrent Inference for YuE2</h1>

<p align="center"><strong>Same YuE2 model. 1.68× faster per song. 3.31× more songs per GPU.</strong></p>

<p align="center">
  <a href="#environment">🧰 Environment</a> ·
  <a href="#deploy-the-accelerated-service">🚀 Deploy</a> ·
  <a href="#configuration-reference">⚙️ Configuration</a> ·
  <a href="#about-yue2">🎵 About YuE2</a> ·
  <a href="https://huggingface.co/m-a-p/YuE2-3B">🤗 Weights</a>
</p>

**YuE2-Turbo is an inference acceleration and serving layer for [YuE2](#about-yue2).** It keeps the released `YuE2-3B` weights and the standard generation recipe (BF16, 32 flow-matching steps) and replaces how they are executed: the autoregressive stages run on **vLLM**, all weights stay **resident on the GPU**, acoustic synthesis is **batched across requests**, and an **HTTP job API + browser Studio** turn one GPU into a concurrent song-generation service.

| Scenario | Original YuE2 | YuE2-Turbo | Speedup |
|---|---:|---:|---:|
| Single request (RTF, lower is better) | 0.290 | 0.173 | **1.68×** |
| 4 concurrent requests (system RTF) | 0.317 | 0.096 | **3.31×** |

*Measured on one NVIDIA RTX 5090 (32 GB), PyTorch 2.10.0 + CUDA 12.8, vLLM 0.19.0, BF16, 32 ODE steps, after warmup. RTF = seconds of compute per second of generated audio: at RTF 0.17, a 60-second song takes about 10 seconds. Single request: 3 songs × 3 repeats. Concurrent: the same 4 requests per wave, 3 waves, wall time ÷ total audio generated. Reproduce with [`yue2-benchmark`](#reproduce-the-benchmark).*

The accelerated path does not lose quality on the [WildSongBench](https://huggingface.co/datasets/m-a-p/WildSongBench) standard protocol. Each of 192 prompts is generated twice; the lower-PER candidate is scored against the published YuE2 row:

| Metric | Original YuE2 | YuE2-Turbo |
|---|---:|---:|
| SongBench Avg ↑ | 6.7316 | 6.7623 |
| MuLan ↑ | 0.5068 | 0.5069 |
| AllMusicCaps ↑ | 0.4054 | 0.4087 |
| PER ↓ | 8.44% | 8.18% |

*One RTX 5090, vLLM with 4-way concurrency, BF16, 32 ODE steps, same released weights and generation recipe.*

## Environment

| Requirement | Tested / recommended |
|---|---|
| OS | Linux x86_64 |
| GPU | NVIDIA with BF16 support. Defaults are tuned for a **32 GB RTX 5090**; on 24 GB cards lower `YUE2_AR_CONCURRENCY` / `YUE2_VLLM_MAX_NUM_SEQS` or set `YUE2_RESIDENT_MODELS=false` and validate |
| Driver | Supports CUDA 12.8 |
| Python | 3.11 or 3.12 |
| PyTorch | 2.10.0 (`cu128` wheels) |
| vLLM / Triton | 0.19.0 / 3.6.0 (installed by the `server` extra) |
| Disk | ~8 GB for `YuE2-3B` + `YuE2-Vae`, plus a few GB for the derived vLLM AR checkpoint cache |

GPU memory at rest with the default configuration is about 18 GB (PyTorch MoT + VAE ≈ 8 GB, vLLM engine ≈ 10 GB); peaks under 4-way load stay under 25 GB on a 32 GB card.

## Deploy the accelerated service

### 1. Install

```bash
git clone <this repository> YuE2-Turbo
cd YuE2-Turbo
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install '.[server]'
```

The `server` extra pulls in FastAPI, Uvicorn, vLLM, and Triton. Weights download from Hugging Face on first start; set `YUE2_MODEL` / `YUE2_VAE` to local directories and `YUE2_LOCAL_FILES_ONLY=true` for offline hosts.

### 2. Configure

```bash
export YUE2_API_KEY="$(openssl rand -hex 32)"   # required, ≥16 chars; share only with callers
export YUE2_DATA_DIR="$PWD/outputs/service"      # SQLite job store + audio/score artifacts
export CUDA_VISIBLE_DEVICES=0                    # one service process per GPU; use an index, not a UUID
```

The defaults already select the accelerated path (`YUE2_BACKEND=vllm`, `YUE2_RESIDENT_MODELS=true`, `YUE2_AR_CONCURRENCY=4`, `YUE2_NAR_BATCH_SIZE=2`, `YUE2_AR_NAR_OVERLAP=true`). See the [configuration reference](#configuration-reference) to tune them.

### 3. Start the API

```bash
yue2-serve            # binds 127.0.0.1:8000 by default (YUE2_HOST / YUE2_PORT)
```

Startup loads the vLLM engine, the PyTorch MoT and VAE, then runs a short warmup song so the first real request is already fast. Watch readiness:

```bash
curl -i http://127.0.0.1:8000/health/ready   # 503 while loading, 200 when ready
```

Interactive OpenAPI docs are at `http://127.0.0.1:8000/docs` (click **Authorize** and paste the API key).

### 4. Start the Studio Web UI (optional)

```bash
YUE2_UPSTREAM=http://127.0.0.1:8000 yue2-web   # serves http://0.0.0.0:8016
```

The Studio is a single-page creation console: lyrics, style, composition mode, optional ABC score, an optional source-audio upload for covers, live progress, playback, and downloads. The browser sends its own bearer key on every call; the gateway never stores or embeds it. Put it behind HTTPS before exposing it beyond a trusted network.

### 5. Call the API

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -H "Authorization: Bearer $YUE2_API_KEY" \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: order-0001' \
  -d '{"style":"Mandarin, piano pop, warm vocal","lyrics":"[Verse]\n晨光落在窗边\n[Chorus]\n让歌声陪伴你","cot":"full","seed":42}'
```

The service answers `202` with a job `id`; poll and download:

```bash
JOB=<id>
curl -H "Authorization: Bearer $YUE2_API_KEY" http://127.0.0.1:8000/v1/jobs/$JOB
curl -H "Authorization: Bearer $YUE2_API_KEY" http://127.0.0.1:8000/v1/jobs/$JOB/audio -o song.flac
curl -H "Authorization: Bearer $YUE2_API_KEY" http://127.0.0.1:8000/v1/jobs/$JOB/score -o score.abc
curl -X POST -H "Authorization: Bearer $YUE2_API_KEY" http://127.0.0.1:8000/v1/jobs/$JOB/cancel
```

| Endpoint | Purpose |
|---|---|
| `GET /health/live`, `GET /health/ready` | Liveness / readiness (public) |
| `POST /v1/jobs` | Submit a song (`n=2` for two seeds at once); same `Idempotency-Key` + same body returns the original job |
| `GET /v1/jobs/{id}` | Status: `queued → running → succeeded / truncated / failed / cancelled`, stage, token counts, timings |
| `POST /v1/jobs/{id}/cancel` | Cancel at the next token / ODE step / decode chunk boundary |
| `GET /v1/jobs/{id}/audio`, `/score` | Download FLAC and ABC |
| `POST /v1/covers` | Cover: transcribe an uploaded song to a melody, then generate (`YUE2_SHEETSAGE_DEVICE` must not be `off`) |

Queue full returns `429` with `Retry-After`; not ready returns `503`. Finished artifacts are kept for 24 h or 5 GiB per data directory by default. Requests with `cfg_scale ≠ 1` or `cot="off"` automatically use the original PyTorch path, so every request type is supported.

### Cover a recording

Covers are on by default. `YUE2_SHEETSAGE_DEVICE` defaults to `auto`. [SheetSage2](https://huggingface.co/m-a-p/SheetSage2) loads on the first cover, so ordinary generation does not spend its memory. It turns the upload into a chord-free melody ABC; YuE2 then realizes that melody with `cot=melody` in the requested style and lyrics. It does not clone the original singer.

`auto` keeps the transcriber on GPU in BF16 only when at least `YUE2_SHEETSAGE_MIN_FREE_GIB` (default 8) is free **after** YuE2 is resident; otherwise it stays on CPU in FP32. A 24 GB card that is already running the resident service usually takes the CPU path. CPU transcription of a full song can take several minutes and still counts against `YUE2_TASK_TIMEOUT_SECONDS`. `ffmpeg` must be on `PATH`. The `server` extra installs the Python packages SheetSage2 imports (`mir_eval`, `pretty_midi`, `mido`, `scipy`). Set the variable to `off` to disable covers.

```bash
curl -X POST http://127.0.0.1:8000/v1/covers \
  -H "Authorization: Bearer $YUE2_API_KEY" \
  -F "audio=@song.mp3" \
  -F "style=Mandarin, piano pop, warm vocal" \
  -F "lyrics=[Verse]
晨光落在窗边
[Chorus]
让歌声陪伴你" \
  -F "seed=42"
```

The same job API returns the FLAC and the melody ABC. Studio exposes this as “上传歌曲翻唱”. The gateway allows a 40 MB body on `POST /v1/covers` only. With `YUE2_SHEETSAGE_DEVICE=off` the endpoint returns 503 and does not store the audio.

From the command line, transcription runs first and the SheetSage2 weights are released before YuE2 loads, so the two models do not stay on the GPU together:

```bash
yue2 cover --audio song.mp3 --request cover-request.json --sheetsage-device auto --output outputs/cover
```

### Configuration reference

All settings are environment variables prefixed `YUE2_`.

| Variable | Default | Meaning |
|---|---|---|
| `YUE2_BACKEND` | `vllm` | `vllm` (accelerated), `torch` (original CUDA-graph path), `torch-eager` |
| `YUE2_RESIDENT_MODELS` | `true` | Keep MoT and VAE on the GPU between songs |
| `YUE2_AR_CONCURRENCY` | `4` | Requests submitted to vLLM simultaneously (≤ `YUE2_VLLM_MAX_NUM_SEQS`) |
| `YUE2_VLLM_MAX_NUM_SEQS` | `4` | vLLM scheduler capacity |
| `YUE2_VLLM_MAX_NUM_BATCHED_TOKENS` | `8192` | Tokens per vLLM scheduling step (chunked prefill) |
| `YUE2_VLLM_GPU_MEMORY_UTILIZATION` | `0.30` | Fraction of the GPU reserved for the vLLM engine (weights + KV cache) |
| `YUE2_NAR_BATCH_SIZE` | `2` | Songs synthesized together in NAR (max 2) |
| `YUE2_AR_NAR_OVERLAP` | `true` | Prefetch the next AR wave while the current wave runs NAR/VAE |
| `YUE2_AR_BATCH_WAIT_MS` | `50` | Short window to coalesce simultaneous submissions |
| `YUE2_MEMORY_BUDGET_GIB` | `30` | Admission budget for PyTorch allocations |
| `YUE2_ODE_STEPS` | `32` | Flow-matching steps; lowering trades quality for NAR speed |
| `YUE2_MAX_PENDING` | `16` | Queue depth before `429` |
| `YUE2_TASK_TIMEOUT_SECONDS` | `1200` | Per-job execution timeout (excludes queueing) |
| `YUE2_ARTIFACT_RETENTION_SECONDS` / `YUE2_ARTIFACT_MAX_GIB` | `86400` / `5` | Artifact cleanup policy |
| `YUE2_WARMUP` | `true` | Run a short warmup song at startup |
| `YUE2_MODEL` / `YUE2_VAE` / `YUE2_LOCAL_FILES_ONLY` | HF ids / `false` | Model sources |
| `YUE2_SHEETSAGE_DEVICE` | `auto` | Cover transcription: `auto` (GPU when free memory is enough, otherwise CPU), `cpu`, `cuda`, or `off` |
| `YUE2_SHEETSAGE_MIN_FREE_GIB` | `8` | Free GPU memory `auto` requires before placing SheetSage2 on GPU |
| `YUE2_SHEETSAGE` / `YUE2_SHEETSAGE_REVISION` | `m-a-p/SheetSage2` / unset | SheetSage2 snapshot used when covers are enabled |

Setting `YUE2_BACKEND=torch YUE2_RESIDENT_MODELS=false YUE2_AR_CONCURRENCY=1 YUE2_NAR_BATCH_SIZE=1 YUE2_AR_NAR_OVERLAP=false` reproduces the original execution behavior behind the same API; this is the "Original YuE2" baseline in the table above.

### Multiple GPUs

Run one `yue2-serve` process per GPU with a distinct `CUDA_VISIBLE_DEVICES`, `YUE2_PORT`, and `YUE2_DATA_DIR`, and load-balance in front of them. Do not run several Uvicorn workers or several processes against the same data directory; the service holds a process lock on it.

### Reproduce the benchmark

```bash
export CUDA_VISIBLE_DEVICES=0            # a GPU with no other workload
yue2-benchmark --check
yue2-benchmark --requests examples/benchmark-requests.jsonl \
  --profiles reference vllm --warmup 1 --repeats 3 \
  --output outputs/benchmark-rtx5090
```

`reference` is the original path, `vllm` the accelerated one. `report.json` records per-stage timings, RTF, token throughput, peak memory, and model hashes for every run. For the concurrent figure, submit the same request set to a running `yue2-serve` at concurrency 1, 2, and 4 and divide each wave's wall time by its total generated audio.

## About YuE2

YuE2-Turbo builds on **YuE2** by the M-A-P community. For the model, demos, evaluation, covers, editing, and agent skill, see the original repository:

**[github.com/multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE)** · [🤗 YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) · [🤗 YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) · [Demo page](https://map-yue2.github.io/)

## License

Code in this repository is licensed under **[Apache 2.0](LICENSE)**. YuE2 model weights are separately licensed under **[CC BY-NC 4.0](MODEL_LICENSE)**; third-party components retain their [original licenses](THIRD_PARTY_NOTICES.md).

## Acknowledgments

Thanks to the [Linux.Do](https://linux.do)
