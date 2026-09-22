<p align="center">
  <img src="assets/yue2-turbo-logo.png" alt="YuE2 Turbo" width="460" />
</p>

<p align="center">
  <a href="README.md">English</a> | <b>中文</b>
</p>

<h1 align="center">YuE2-Turbo: YuE2 高性能并发推理服务</h1>

<p align="center"><strong>同样的 YuE2 模型。单曲生成提速 1.68×，单卡吞吐提升 3.31×。</strong></p>

<p align="center">
  <a href="#环境要求">🧰 环境要求</a> ·
  <a href="#部署加速服务">🚀 部署服务</a> ·
  <a href="#配置说明">⚙️ 配置说明</a> ·
  <a href="#关于-yue2">🎵 关于 YuE2</a> ·
  <a href="https://huggingface.co/m-a-p/YuE2-3B">🤗 权重下载</a>
</p>

**YuE2-Turbo 是面向 [YuE2](#关于-yue2) 的推理加速与生产服务化方案。** 在保持官方发布的 `YuE2-3B` 权重与标准生成规格（BF16、32 步 flow-matching 采样）完全一致的前提下，重构了底层执行架构：自回归（AR）阶段采用 **vLLM** 加速，全流程权重 **常驻 GPU 显存**，非自回归声学阶段（NAR）支持 **跨请求批处理**，并提供开箱即用的 **HTTP 异步任务 API + 网页版 Studio Web UI**，单张 GPU 即可高效承载多用户并发生成。

| 场景 | 原版 YuE2 | YuE2-Turbo | 加速比 |
|---|---:|---:|---:|
| 单请求 (RTF，越低越好) | 0.290 | 0.173 | **1.68×** |
| 4 并发请求 (系统 RTF) | 0.317 | 0.096 | **3.31×** |

*测试环境：单卡 NVIDIA RTX 5090 (32 GB)，PyTorch 2.10.0 + CUDA 12.8，vLLM 0.19.0，BF16，32 步 ODE，充分预热后测得。RTF = 生成每秒音频所需的计算时间（秒）：RTF 0.17 意味着生成一段 60 秒歌曲仅需约 10 秒。单请求指标基于 3 首歌曲 × 3 次重复取平均；并发指标基于每波 4 个请求 × 3 波测得，系统 RTF = 单波耗时 ÷ 生成的总音频时长。可使用 [`yue2-benchmark`](#复现性能评测) 完整复现。*

加速路径在 [WildSongBench](https://huggingface.co/datasets/m-a-p/WildSongBench) 标准协议上没有掉点。192 题各生成 2 首，取得分更低的 PER 候选，对照公布的 YuE2 标准分数：

| 指标 | 原版 YuE2 | YuE2-Turbo |
|---|---:|---:|
| SongBench Avg ↑ | 6.7316 | 6.7623 |
| MuLan ↑ | 0.5068 | 0.5069 |
| AllMusicCaps ↑ | 0.4054 | 0.4087 |
| PER ↓ | 8.44% | 8.18% |

*单卡 RTX 5090，vLLM 4 路并发，BF16、32 步 ODE，与官方同一套权重和生成规格。*

## 环境要求

| 依赖项 | 测试验证 / 推荐配置 |
|---|---|
| 操作系统 | Linux x86_64 |
| GPU | 支持 BF16 的 NVIDIA 显卡。默认参数针对 **32 GB RTX 5090** 优化；在 24 GB 显卡上建议调小 `YUE2_AR_CONCURRENCY` / `YUE2_VLLM_MAX_NUM_SEQS` 或设置 `YUE2_RESIDENT_MODELS=false` |
| 驱动 | 支持 CUDA 12.8 |
| Python | 3.11 或 3.12 |
| PyTorch | 2.10.0 (`cu128` wheels) |
| vLLM / Triton | 0.19.0 / 3.6.0（包含在 `server` 依赖中） |
| 磁盘空间 | `YuE2-3B` + `YuE2-Vae` 约需 8 GB，另需数 GB 存放 vLLM 转换后的 AR 权重缓存 |

默认配置下服务静默显存占用约 18 GB（PyTorch MoT + VAE 约 8 GB，vLLM 引擎约 10 GB）；在 32 GB 显卡上应对 4 路并发时，峰值显存保持在 25 GB 以内。

## 部署加速服务

### 1. 安装

```bash
git clone <this repository> YuE2-Turbo
cd YuE2-Turbo
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install '.[server]'
```

`server` 依赖包含了 FastAPI、Uvicorn、vLLM 与 Triton。初次启动会自动从 Hugging Face 下载权重；如需离线运行，可设置 `YUE2_MODEL` / `YUE2_VAE` 指向本地目录并指定 `YUE2_LOCAL_FILES_ONLY=true`。

### 2. 配置环境变量

```bash
export YUE2_API_KEY="$(openssl rand -hex 32)"   # 必填，API 密钥（不少于 16 字符）
export YUE2_DATA_DIR="$PWD/outputs/service"      # SQLite 任务库与音频/乐谱产物存储目录
export CUDA_VISIBLE_DEVICES=0                    # 每个服务进程独占一张卡；使用卡号索引，勿用 UUID
```

默认设置已经全量开启加速特性（`YUE2_BACKEND=vllm`、`YUE2_RESIDENT_MODELS=true`、`YUE2_AR_CONCURRENCY=4`、`YUE2_NAR_BATCH_SIZE=2`、`YUE2_AR_NAR_OVERLAP=true`）。如需微调，请参考 [配置说明](#配置说明)。

### 3. 启动 API 服务

```bash
yue2-serve            # 默认监听 127.0.0.1:8000（可通过 YUE2_HOST / YUE2_PORT 修改）
```

服务启动时会依次加载 vLLM 引擎、PyTorch MoT 与 VAE，并自动执行一段预热生成，确保后续真实请求直接达到最优速度。检查就绪状态：

```bash
curl -i http://127.0.0.1:8000/health/ready   # 加载阶段返回 503，就绪后返回 200
```

可访问 `http://127.0.0.1:8000/docs` 查看交互式 OpenAPI 接口文档（点击 **Authorize** 输入配置的 API Key 即可调试）。

### 4. Web UI 启动！

```bash
YUE2_UPSTREAM=http://127.0.0.1:8000 yue2-web   # 默认监听 http://0.0.0.0:8016
```

Studio 提供了可视化的单页音乐创作工作台：支持歌词与风格输入、作曲模式切换、ABC 乐谱预览、可选的原曲上传翻唱、实时生成进度反馈、音频播放与结果下载。浏览器在请求时携带用户填入的 Bearer Token，网关本身不硬编码或持久化密钥。若在非受信网络暴露，建议在前端配置 HTTPS。

### 5. 调用 API

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -H "Authorization: Bearer $YUE2_API_KEY" \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: order-0001' \
  -d '{"style":"Mandarin, piano pop, warm vocal","lyrics":"[Verse]\n晨光落在窗边\n[Chorus]\n让歌声陪伴你","cot":"full","seed":42}'
```

接口返回 `202 Accepted` 以及任务 `id`，轮询任务并下载产物：

```bash
JOB=<id>
curl -H "Authorization: Bearer $YUE2_API_KEY" http://127.0.0.1:8000/v1/jobs/$JOB
curl -H "Authorization: Bearer $YUE2_API_KEY" http://127.0.0.1:8000/v1/jobs/$JOB/audio -o song.flac
curl -H "Authorization: Bearer $YUE2_API_KEY" http://127.0.0.1:8000/v1/jobs/$JOB/score -o score.abc
curl -X POST -H "Authorization: Bearer $YUE2_API_KEY" http://127.0.0.1:8000/v1/jobs/$JOB/cancel
```

| 接口 | 用途 |
|---|---|
| `GET /health/live`, `GET /health/ready` | 存活探针 / 就绪探针（无需鉴权） |
| `POST /v1/jobs` | 提交生成任务（支持 `n=2` 同时生成多个随机 seed）；相同 `Idempotency-Key` + 相同请求体自动幂等返回原任务 |
| `GET /v1/jobs/{id}` | 查询任务详情：状态（`queued → running → succeeded / truncated / failed / cancelled`）、阶段、Token 统计、各阶段耗时 |
| `POST /v1/jobs/{id}/cancel` | 取消任务（在下一个 Token / ODE 步 / VAE 分块边界处快速退出） |
| `GET /v1/jobs/{id}/audio`, `/score` | 下载 FLAC 音频或 ABC 乐谱 |
| `POST /v1/covers` | 翻唱：把上传的歌曲转成旋律后再生成（`YUE2_SHEETSAGE_DEVICE` 不能为 `off`） |

队列已满时返回 `429`（附带 `Retry-After` 头）；服务未就绪时返回 `503`。生成产物默认按每个数据目录保留 24 小时或最多 5 GiB。若请求需要 CFG（`cfg_scale ≠ 1`）或关闭思维链（`cot="off"`），服务会自动平滑回退至原版 PyTorch 分支执行，无需人工干预。

### 翻唱

翻唱默认开启。`YUE2_SHEETSAGE_DEVICE` 默认是 `auto`。[SheetSage2](https://huggingface.co/m-a-p/SheetSage2) 只在第一次翻唱时加载，普通生成不会占用它的显存。它把上传的歌曲转成不含和弦的旋律 ABC，YuE2 再用 `cot=melody` 按给定风格和歌词生成。它不克隆原唱音色。

`auto` 会在 YuE2 已经常驻之后看剩余显存：至少还有 `YUE2_SHEETSAGE_MIN_FREE_GIB`（默认 8 GiB）才把转谱放在 GPU 上并以 BF16 运行，否则留在 CPU 上用 FP32。24 GB 卡在生成服务常驻后通常走 CPU。CPU 转一首完整歌曲可能要数分钟，这段时间同样计入 `YUE2_TASK_TIMEOUT_SECONDS`。系统 `PATH` 里需要有 `ffmpeg`。`server` 附加依赖已经包含 SheetSage2 导入所需的 Python 包（`mir_eval`、`pretty_midi`、`mido`、`scipy`）。显式设为 `off` 可关闭翻唱。

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

之后仍用原来的任务接口下载 FLAC 和旋律谱。Studio 里的入口是「上传歌曲翻唱」。网关只对 `POST /v1/covers` 放宽到 40 MB，其它请求仍是 256 KB。`YUE2_SHEETSAGE_DEVICE=off` 时该接口返回 503，并且不会保存音频。

命令行会先转谱，再释放 SheetSage2 的权重，然后才加载 YuE2，避免两个模型同时留在显卡上：

```bash
yue2 cover --audio song.mp3 --request cover-request.json --sheetsage-device auto --output outputs/cover
```

### 配置说明

所有配置均通过 `YUE2_` 前缀的环境变量读取：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `YUE2_BACKEND` | `vllm` | 推理后端：`vllm`（加速路径）、`torch`（原版 CUDA Graph 路径）、`torch-eager` |
| `YUE2_RESIDENT_MODELS` | `true` | MoT 与 VAE 模型常驻显存，避免任务间反复卸载加载 |
| `YUE2_AR_CONCURRENCY` | `4` | 同时提交给 vLLM 并行调度的请求数（需 ≤ `YUE2_VLLM_MAX_NUM_SEQS`） |
| `YUE2_VLLM_MAX_NUM_SEQS` | `4` | vLLM 调度器最大并发序列数 |
| `YUE2_VLLM_MAX_NUM_BATCHED_TOKENS` | `8192` | vLLM 单次调度最大 Token 数（chunked prefill） |
| `YUE2_VLLM_GPU_MEMORY_UTILIZATION` | `0.30` | 预留给 vLLM 引擎（权重 + KV Cache）的显存比例 |
| `YUE2_NAR_BATCH_SIZE` | `2` | NAR 声学合成批处理请求数（最大为 2） |
| `YUE2_AR_NAR_OVERLAP` | `true` | 重叠调度：当上一批请求在执行 NAR/VAE 时，提前调度下一批 AR 生成 |
| `YUE2_AR_BATCH_WAIT_MS` | `50` | 批量接收并发请求的微等待窗口（毫秒） |
| `YUE2_MEMORY_BUDGET_GIB` | `30` | PyTorch 显存分配配额预算 |
| `YUE2_ODE_STEPS` | `32` | Flow-matching 采样步数；减小可进一步提升 NAR 速度，但对音质略有折损 |
| `YUE2_MAX_PENDING` | `16` | 任务排队最大深度，超出返回 `429` |
| `YUE2_TASK_TIMEOUT_SECONDS` | `1200` | 单任务执行超时时间（秒，不计排队时间） |
| `YUE2_ARTIFACT_RETENTION_SECONDS` / `YUE2_ARTIFACT_MAX_GIB` | `86400` / `5` | 产物自动清理策略（保留时长 / 空间上限） |
| `YUE2_WARMUP` | `true` | 服务启动时自动运行预热任务 |
| `YUE2_MODEL` / `YUE2_VAE` / `YUE2_LOCAL_FILES_ONLY` | HF 仓库 ID / `false` | 模型来源与离线加载模式 |
| `YUE2_SHEETSAGE_DEVICE` | `auto` | 翻唱转谱：`auto`（空闲显存够则 GPU，否则 CPU）、`cpu`、`cuda`，或 `off`（关闭） |
| `YUE2_SHEETSAGE_MIN_FREE_GIB` | `8` | `auto` 把 SheetSage2 放到 GPU 前要求的空闲显存（GiB） |
| `YUE2_SHEETSAGE` / `YUE2_SHEETSAGE_REVISION` | `m-a-p/SheetSage2` / 未设置 | 启用翻唱时加载的 SheetSage2 快照 |

设置 `YUE2_BACKEND=torch YUE2_RESIDENT_MODELS=false YUE2_AR_CONCURRENCY=1 YUE2_NAR_BATCH_SIZE=1 YUE2_AR_NAR_OVERLAP=false` 可在相同 API 接口下完全还原原版执行行为（即上表中“原版 YuE2”基准）。

### 多卡部署

建议每张物理 GPU 单独启动一个 `yue2-serve` 进程，分别配置不同的 `CUDA_VISIBLE_DEVICES`、`YUE2_PORT` 与 `YUE2_DATA_DIR`，并在前端使用 Nginx 或网关做负载均衡。请勿使用多个 Uvicorn worker 或多进程同时操作同一个数据目录（服务会对数据目录加进程锁）。

### 复现性能评测

```bash
export CUDA_VISIBLE_DEVICES=0            # 指定一张无其他负载的测试 GPU
yue2-benchmark --check
yue2-benchmark --requests examples/benchmark-requests.jsonl \
  --profiles reference vllm --warmup 1 --repeats 3 \
  --output outputs/benchmark-rtx5090
```

`reference` 代表原版路径，`vllm` 代表加速路径。测试完成后生成的 `report.json` 会详细记录每次运行的各阶段耗时、RTF、Token 吞吐率、峰值显存与模型哈希。并发测试可通过向运行中的 `yue2-serve` 提交相同请求批次（并发度 1、2、4），并统计整波任务墙钟耗时与生成总音频时长的比值测得。

## 关于 YuE2

YuE2-Turbo 基于 M-A-P 社区发布的 **YuE2** 构建。模型权重、演示试听、评估评测、翻唱/编辑模式与 Agent Skill 等详细介绍，请参阅原仓库：

**[github.com/multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE)** · [🤗 YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) · [🤗 YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) · [Demo 页面](https://map-yue2.github.io/)

## 开源许可

本项目代码遵循 **[Apache 2.0](LICENSE)** 协议开源。YuE2 模型权重独立遵循 **[CC BY-NC 4.0](MODEL_LICENSE)** 非商业许可协议；第三方组件保留其[原始开源许可](THIRD_PARTY_NOTICES.md)。

## 致谢

感谢 [Linux.Do](https://linux.do)
