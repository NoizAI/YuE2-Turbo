# Noiz YuE2 服务与 5090 推理优化

此分支增加 FastAPI 异步服务和显存常驻优化。部署目标：Linux、标准 RTX 5090 32 GB、
通用镜像使用 Python 3.12、PyTorch 2.10.0 CUDA 12.8。ucloud-4 已在独立 Python 3.11 环境完成真实 GPU 验证，见下文实测。
这是单机、单 GPU、单租户服务：持有 API 密钥的调用方可读取本服务中的全部任务。

## 在 5090 上运行

先安装支持 5090 的 NVIDIA 驱动，确认 `nvidia-smi` 正常。
Docker 路径还需要 Docker Compose 与 NVIDIA Container Toolkit。

```bash
git clone https://github.com/NoizAI/YuE.git
cd YuE
git switch codex/fastapi-inference-acceleration
cp .env.example .env
# 编辑 .env：用 openssl rand -hex 32 的结果替换 YUE2_API_KEY。
# YUE2_GPU_DEVICE 指定空闲 GPU 的 UUID（nvidia-smi -L 查看），默认 GPU 0。
# YUE2_PORT 可换空闲端口；默认限制 4 CPU / 48 GiB 内存。
docker compose up --build -d
docker compose logs -f yue2
curl -i http://127.0.0.1:8000/health/ready
```

镜像不含模型权重，首次启动下载到持久化卷并预热。`/health/ready` 返回 200 后接任务，
加载或失败时返回 503；`/health/live` 只表示 HTTP 进程存活。
镜像构建与 CUDA 执行需要在 Linux/amd64 验证，CPU 单测不能替代它。
默认只映射本机端口。跨机器调用时通过业务网关提供 HTTPS 和访问控制。

也可以直接在 Linux 主机安装：

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install '.[server]'
export YUE2_API_KEY="$(openssl rand -hex 32)"
export YUE2_DATA_DIR="$PWD/outputs/service"
yue2-serve
```

`.env` 由 Compose 读取；直接运行时自行导出环境变量。保留密钥给调用方，不要提交到 Git。
可选配置：`YUE2_MODEL` / `YUE2_VAE`、`YUE2_REVISION` / `YUE2_VAE_REVISION` 指定模型和版本；
`YUE2_LOCAL_FILES_ONLY=true` 禁止自动下载。Compose 使用这些可选项时要补入 environment。

## 接口

Swagger：`http://127.0.0.1:8000/docs`，点击 Authorize 输入密钥。
健康检查和 OpenAPI 文档公开，所有业务接口需要 Bearer token。

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -H "Authorization: Bearer $YUE2_API_KEY" \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: my-business-order-001' \
  -d '{"style":"Mandarin, piano pop, warm vocal", "lyrics":"[Verse]\n晨光落在窗边\n我们走向新的一天\n[Chorus]\n让歌声陪伴你", "cot":"full", "seed":42}'
```

返回 202，含 `id`、`status=queued`；Location 指向状态接口。
相同幂等键和相同输入重试返回原任务（200），不同输入返回 409；幂等键不自动过期。
再次生成需用新键，接口每次只生成一个候选。

```bash
JOB_ID=替换为返回的id
curl -H "Authorization: Bearer $YUE2_API_KEY" "http://127.0.0.1:8000/v1/jobs/$JOB_ID"
curl -H "Authorization: Bearer $YUE2_API_KEY" "http://127.0.0.1:8000/v1/jobs/$JOB_ID/audio" -o song.flac
curl -H "Authorization: Bearer $YUE2_API_KEY" "http://127.0.0.1:8000/v1/jobs/$JOB_ID/score" -o score.abc
curl -X POST -H "Authorization: Bearer $YUE2_API_KEY" "http://127.0.0.1:8000/v1/jobs/$JOB_ID/cancel"
```

状态：`queued` → `running` → `succeeded` / `truncated` / `failed` / `cancelled`。
`stage` 为 loading、planning、semantic、synthesis、decode、saving、finished；
`tokens` 为已生成 token 数，不是歌曲完成百分比。时间戳使用 Unix 秒。
结果含音频/可选乐谱地址、采样率、时长、各阶段耗时、实际配置。
`truncated` 明确表示达到 token 上限，音频可下载，但不能当作完整成功。

`cot=off` 不生成乐谱，score 返回 404。翻唱先用 SheetSage2 转谱，再提交 `abc` 文本与
`cot=melody`；此服务不包含原音频转谱。`cfg_scale` 默认沿用上游各模式对应值。
请求体限 256 KiB，文本另有长度验证。若乐谱或生成内容超出模型上下文，任务会失败，不偷偷截短。
队列满返回 429（带 Retry-After），未就绪返回 503。

取消/超时在 token、合成步和解码块之间检查，不能立即抢占已执行的 GPU 运算。
超时默认 1200 秒，从开始执行计时，不含排队；保存期间的取消要等保存结束。

## 运维与恢复

- 一个服务进程独占 GPU；兼容请求的 AR 阶段由 vLLM 并行，NAR 可 FIFO 合批，VAE 仍串行。
- SQLite 保存任务和幂等键；重启继续 queued 任务，原 running 标记 `worker_interrupted`。
- 同一数据目录使用进程锁，禁止多 Uvicorn worker。多 GPU 使用不同 CUDA_VISIBLE_DEVICES、端口和数据目录。
- 推理异常会记录服务端日志并重建模型；HTTP 不暴露原始异常中的内部路径。
- 数据库与 `YUE2_DATA_DIR/artifacts/<id>/` 中的音频、乐谱、生成记录应一起备份。
- 服务每 300 秒自动清理终态任务的本地产物：默认保留 24 小时，并将每个数据目录的
  `artifacts/` 限制在 5 GiB；流量过高时容量上限优先，最旧产物可能提前过期。
  `YUE2_ARTIFACT_RETENTION_SECONDS`、`YUE2_ARTIFACT_MAX_GIB` 和
  `YUE2_ARTIFACT_CLEANUP_INTERVAL_SECONDS` 可调整策略。queued/running 任务不会被清理；
  SQLite 任务和幂等记录保留，已过期产物的下载接口返回 404。
- 仍需监控整个磁盘；模型缓存、基准输出等不在服务清理范围内。未提供多租户隔离、对象存储或分布式队列。

## 加速配置

默认服务：`vLLM AR + max_num_seqs=4 + AR concurrency=4 + NAR batch=2 + AR/NAR overlap + BF16 + 32 步`。
vLLM 只执行乐谱和 semantic token 两个 AR 阶段；NAR 声学生成和 VAE 解码使用 PyTorch。
服务启动时加载一次 vLLM 引擎，并保持到服务退出，不再在每首歌进入 NAR 时销毁。
服务最多提前准备一波 AR：当前波按原始 FIFO 顺序执行 padded NAR batch 和逐条 VAE decode 时，
后台可认领下一波并提交到同一个 vLLM 调度器。prepared 结果不能越过队头解码或保存。
`cot=off`、CFG fallback 和非 vLLM 请求仍是独占 barrier，不允许后续任务跨越。

默认并行参数针对当前 32 GB RTX 5090 设置为 4。`YUE2_VLLM_MAX_NUM_SEQS` 控制引擎容量，
`YUE2_AR_CONCURRENCY` 控制服务同时提交的 AR 请求，后者不能大于前者。提高这两个值会线性增加
KV 需求；4 路满 24576 上下文理论上约需 10.5 GiB KV。
`YUE2_VLLM_MAX_NUM_BATCHED_TOKENS=8192` 控制一次调度可处理的 token 数，chunked prefill
仍保持开启；它提高长 prompt 的 prefill 上限，但会增加引擎 profiling 的峰值和显存需求。
`YUE2_AR_BATCH_WAIT_MS=50` 给同时到达的请求一个很短的合批窗口；低延迟优先时可调低，
吞吐优先时可在压测后适当调高。
`YUE2_AR_NAR_OVERLAP=true` 启用跨请求流水线；设为 `false` 可恢复原 AR barrier 行为。
`YUE2_NAR_BATCH_SIZE=2` 控制 NAR 窗口且当前最大只能设为 2。请求不会按长度排序或分桶，每行保留自己的
prefix、RoPE 位置、noise/seed 和有效长度；32 步 midpoint ODE 不变，每步仍执行两次 velocity，
但 QKV/MLP 按 batch 合并。不同长度的 attention 按行裁掉 padding，以避免 masked GQA 退化成
显式二次方 attention 矩阵。显存准入保留 4 GiB 安全余量；2 路不满足或首次运行时 OOM 时，
服务暂停新 AR、等待在途 AR 释放运行时资源后重试，仍不满足才按 FIFO 降为逐条执行。

服务启动预加载 vLLM、完整 PyTorch MoT、VAE 和短预热，将初始化移出正式任务。
`YUE2_WARMUP=false` 跳过短预热，但保留 vLLM 引擎预加载。
32 GB 部署默认保持 `YUE2_RESIDENT_MODELS=true`，避免每首歌在 NAR/VAE 阶段搬运权重。
NAR 的 acoustic prefix prefill 仍使用 MoT 的 embedding 和 AR 层，因此不能只常驻纯 NAR 子集；
这会与 vLLM AR 重复保存部分权重，不适合作为较小显存设备的无条件默认。
`YUE2_VLLM_GPU_MEMORY_UTILIZATION=0.30` 将 vLLM 执行器限制在整卡约 30% 的显存目标内。
该比例不是整个 vLLM + MoT + VAE 流水线的硬上限；长请求仍可能被抢占或重计算，必须以真实
并发压测确认总峰值和安全余量。

GPU 5（RTX 5090、未启用 NVIDIA MPS）以 NAR batch=2 各压测 8 路短歌和长歌，barrier 与 overlap
均为 16/16 成功、无 OOM，NVML 峰值分别为 24.22/24.22 GiB。短歌 wall time 为
53.06/53.66 秒，长歌为 163.33/166.68 秒；流水线确实发生重叠，但该环境主要体现为两个 CUDA
context 争用，吞吐分别下降 1.1% 和 2.0%。按请求默认保留 overlap 支持，可通过配置关闭以获得
该机器当前更高的吞吐。0.30 配置在 `max_num_batched_tokens=8192` 下提供约 4.4 GiB vLLM KV cache
（41,200 tokens），仍不足以保证 4 路 AR 请求同时占满 24,576 上下文。

实验选项需要压测、试听后启用：

- `YUE2_ODE_STEPS=16`：减少合成迭代，可能影响音质；只加速合成阶段，不代表整首快一倍。
- `YUE2_QUANTIZATION=fp8`：仅支持 torch 后端，会退出 CUDA Graph 路径，可能更慢；默认关闭。
- `YUE2_BACKEND=torch`：回退到单请求 CUDA Graph，可与 `YUE2_RESIDENT_MODELS=true` 配合。

`cot=off` 或显式 `cfg_scale != 1` 仍会回退 torch AR，以保留历史 CFG/采样语义；这类请求不进入
并行 vLLM 批次。默认 full/melody、默认 CFG=1 的请求使用并行 vLLM AR。

## 真实 GPU 对比

选择空闲且专供本次测试的 GPU，在主机虚拟环境中运行：

```bash
export CUDA_VISIBLE_DEVICES=替换为空闲GPU的UUID
yue2-benchmark --check
yue2-benchmark --requests examples/benchmark-requests.jsonl \
  --profiles reference resident --warmup 1 --repeats 3 \
  --output outputs/benchmark-5090
```

reference = 上游搬运策略＋原 CUDA Graph；resident = 常驻模式。
两组使用相同请求、种子和默认 32 步；加载和整曲预热单独记录。输出目录必须是新目录。
每个候选保留完整音频及中间结果，report.json 包含：

- 环境、显卡、模型身份、实际配置与实际后端。
- 每次生成/保存耗时、歌曲长度、RTF（生成秒数÷音频秒数）、分阶段时间。
- PyTorch 已分配/保留显存峰值（不等同于整张卡的 NVML 峰值）。
- 失败、截断数量；完整成功任务 p50/p95（最近秩法），不计加载、预热、保存、排队、上传。

三条示例只供冒烟，正式评估应使用 20～30 条实际请求，覆盖中英文、长短歌词、外部乐谱和 CFG，
固定模型版本并重复。比较速度时检查输出时长、截断标记，不能把歌变短算作加速；
再试听歌词准确度、旋律、噪声和结尾。CPU 小模型等价性测试不证明完整模型的 GPU 性能和质量。

## 开发验证

```bash
python -m pip install '.[server,test]'
python -m pytest -q
```

接口测试使用模拟模型；常驻模式测试使用真实小型 VAE 检查输出一致和解码中断。
完整模型与 CUDA 专项测试需要 GPU。代码沿用 Apache 2.0；权重仍受 CC BY-NC 4.0 约束。

## 2026-09-14：ucloud-4 GPU 5 实测

服务地址：服务器本机 `http://127.0.0.1:8015`，容器 `noiz-yue-gpu5`。
部署目录 `/data/noiz-yue`，具体操作见 [部署说明](../deploy/ucloud4/README.md)。
Docker Hub 不可达，因此使用服务器已有镜像的固定 SHA256 作为只读基础，挂载独立虚拟环境；
本次没有完成根目录 Dockerfile 的完整构建验证。

固定 32 步、BF16，单 GPU、4 CPU 限额；两条原创新歌词，每个模式两次正式生成，另各预热一次：

- 69.96 秒中文歌曲：reference 平均 20.94 秒，resident 平均 16.69 秒，耗时减少 20.3%。
- 242.24 秒英文歌曲：reference 平均 66.17 秒，resident 平均 61.88 秒，耗时减少 6.5%。
- 四组配对的音频、前缀、语义 token、声学 latent 文件 SHA-256 均相同；八次正式生成均完整成功。
- resident 的 PyTorch 已分配显存峰值约 9.61 GiB；这不是整卡 NVML 显存峰值。
- 真实 HTTP 测试成功生成同一条中文歌曲；提交到轮询发现完成约 18.1 秒（包含保存与最多 2 秒轮询间隔）。
  鉴权 401、幂等提交、排队取消、FLAC 和 ABC 下载均通过。
- GPU 测试：209 passed、2 skipped、30 subtests passed。跳过可选 vLLM 与未安装的原始发布包对照。
- 原有七个 worker 的 PID 和启动时间在测试前后完全一致。所有新增 GPU 工作只用指定 GPU 5。

原始数字见 [实测数据](benchmarks/ucloud4-5090.json)。这是两条样本的测量，不是生产流量的 p95 或所有歌曲的保证。
完整测试记录与可试听音频保存在本次交付目录 `/Users/jishenwei/workFile/2026/0914yue2`。
常驻优化每首主要减少约 4 秒模型搬运开销，所以短歌曲的相对收益更明显。

## Experimental shared-model batching

The opt-in Python pipeline provides `generate_batch(requests)` with conservative
CUDA memory admission. It batches autoregressive forwards and keeps synthesis
and decoding sequential. See [variable-size batching and sweep](batching.md)
and the historical [batch=2 experiment](batch2.md). The HTTP worker still
executes one job at a time; this prototype does not alter its queue policy.

## Two candidates per request

`POST /v1/jobs` accepts `n=2` (default `n=1` retains the original response).
The pair is admitted atomically, needs two free pending slots, and uses the supplied
seed and `(seed + 1) % 2**63`. One group ID is returned with two ordered `candidates`.
The worker generates each candidate independently using the existing concurrent
inference path; model weights are unchanged.

Poll `GET /v1/jobs/{group_id}` for both candidate snapshots, including their individual
progress, error and result fields. `result.outputs` lists completed outputs and can
contain fewer than two entries until both finish. Each candidate has its own job ID
and audio/score URLs; group IDs do not have an audio file. Group states include
`partial_failed` when only some candidates succeeded. `truncated` remains distinct
from complete success. Cancelling a group cancels its unfinished candidates; an
individual candidate can also be cancelled by ID. Finished results remain available.

The group and children are durable in SQLite, and share one idempotency key at the
group level. Retrying the same input/key returns the same two candidates; changing
n or the generation input with that key returns 409. Pair requests use two queue
slots and compute two outputs; latency is not guaranteed to equal one generation.
