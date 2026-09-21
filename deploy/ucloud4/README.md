# ucloud-4 隔离部署

仅面向当前服务器；通用新环境请使用仓库根目录 Dockerfile / compose.yaml。

- 项目目录：`/data/noiz-yue`，容器内挂载为 `/opt/noiz-yue`。
- 使用 GPU 5 的固定 UUID，避免误用其他卡。所有启动脚本会拒绝已有计算进程的 GPU。
- 使用已有镜像的不可变 SHA256 作为只读基础，覆盖原镜像入口；本项目依赖安装到独立 `venv/`，不修改原镜像。
- 本项目 Python 3.11、PyTorch 2.10.0 + CUDA 12.8。专用缓存位于 `torch-cache/`、`triton-cache/`。
- 测试和 GPU 服务串行启动，不同时运行。GPU 容器最多 4 CPU / 48 GiB 内存，
  仅映射本机 8015；独立网页网关可映射公网 80 端口。
- 下载和完整性校验完成标记为 `models/download-complete.json`；模型版本记录为 `models/revisions.json`。
- `service.env` 保存密钥与配置，权限 0600；勿提交或在日志中打印。

服务器上的执行顺序：

```bash
bash /data/noiz-yue/noiz-yue-gpu-test.sh
sudo docker logs -f noiz-yue-gpu-tests
# 检查测试退出码为 0，随后执行：
bash /data/noiz-yue/noiz-yue-bench.sh
sudo docker logs -f noiz-yue-benchmark-gpu5
# 检查结果并确认基准容器退出，随后执行：
bash /data/noiz-yue/noiz-yue-launch.sh
sudo docker logs -f noiz-yue-gpu5
curl -i http://127.0.0.1:8015/health/ready
python3 /data/noiz-yue/api-smoke.py
# 可选：启动不持有 API 密钥的公网 Studio 网关
bash /data/noiz-yue/source/deploy/ucloud4/launch-web.sh
curl http://127.0.0.1/console/status
```

脚本文件名对应仓库中的 `gpu-test.sh`、`bench.sh`、`launch.sh` 和
`launch-web.sh`。
已存在同名容器时脚本会失败；先检查其状态，仅在确认属于本项目且已退出后移除对应容器。
不要停止其他 worker，不要修改 Docker daemon、宿主驱动或共享 Python 环境。

本地访问：

```bash
ssh -N -L 8015:127.0.0.1:8015 ucloud-4
```

打开 `http://127.0.0.1:8015/docs`，业务请求使用 `service.env` 中的 Bearer API 密钥。
基准报告与接口验证输出在服务器 `results/` 目录。

公网 Studio 不嵌入、不保存服务端密钥，浏览器输入的 Bearer 密钥仅透传给同机 GPU 服务。
直接开放 80 端口只适合受控试用；生产环境必须在网关前配置域名与 HTTPS，否则密钥会以明文传输。
