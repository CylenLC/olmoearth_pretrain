# MLflow 实验跟踪

基于官方 `upstream/main`：`6583a536c428ad004bb3c223add0576087d67e92`。
OLMo-core 保持 2.6.0，锁定的 PyTorch 保持 2.9.1。新增可选依赖为
`mlflow-skinny==3.18.0`、psutil 和 nvidia-ml-py；本仓库不部署服务端。

## 安装与连接

已有兼容训练环境：

```bash
pip install -e '.[mlflow]'
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
curl --fail "$MLFLOW_TRACKING_URI/health"
```

全新环境使用：

```bash
uv sync --locked --extra training --extra beaker --extra eval --extra mlflow
source .venv/bin/activate
```

官方入口即使禁用评测也需要导入评测依赖。本地验证环境因 class-registry
依赖 pkg_resources，需要 `setuptools<81`；按实际环境处理。

服务端位于另一台 ECS 时，在训练机器运行隧道（替换 SSH 用户和地址）：

```bash
ssh -N -L 5000:127.0.0.1:5000 USER@ECS_HOST
```

然后继续使用 localhost 地址。认证通过 MLflow 支持的环境变量提供，
例如 `MLFLOW_TRACKING_USERNAME`、`MLFLOW_TRACKING_PASSWORD` 或
`MLFLOW_TRACKING_TOKEN`，不要将凭证写入配置、命令参数或数据集名称。

## ECS GPU 自动验证

要求：可用 NVIDIA CUDA GPU、匹配的驱动和 PyTorch、可用 NVML、上述依赖、
可访问的 Tracking Server 和可写 artifact 存储。服务端需要支持客户端使用的
MLflow Tracking API。脚本会创建真实测试 Run 和少量配置/指标 artifact。

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_VALIDATION_DIR=/tmp/mlflow-validation-$(date +%Y%m%d-%H%M%S)
python scripts/tools/validate_mlflow_ecs.py
```

`MLFLOW_VALIDATION_DIR` 必须尚不存在，防止复用旧 Checkpoint 造成假通过。
默认路径为 `/tmp/olmoearth-mlflow-validation`。凭证环境变量按服务端需要设置。
脚本依次执行真实 Nano 两步训练、恢复至第三步、从两份 Trainer state 核对
Run ID、查询第三步指标及 GPU 利用率/显存指标；再连接本机保留但未监听的端口，
检查 warn 模式仍完成两步训练并保存 Checkpoint。不会停止真实 Tracking Server。
此外，脚本通过仅绑定 127.0.0.1 的本地代理，在第 1 步训练指标成功上传后关闭后续客户端连接，验证训练继续到第 3 步、Run ID 保留且记录降级警告。断连后通过原始地址读取 Run，确认真实服务端仍可用。不会停止服务、修改防火墙或影响其他训练客户端。代理不覆盖 artifact 存储链路；本测试针对 Tracking API 断连。测试 Run 可能因无法发送结束请求而保持 RUNNING，这是故障场景的预期结果。每段日志和 `result.json`
保存在验证目录；失败返回非零退出码，保留证据。

## Nano 单独运行与恢复

在 CUDA 机器、仓库根目录运行：

```bash
python scripts/tools/mlflow_nano_smoke.py --work-dir /tmp/nano-mlflow --mlflow --steps 2
python scripts/tools/mlflow_nano_smoke.py --work-dir /tmp/nano-mlflow --mlflow --steps 3
```

第二条命令自动恢复原 Checkpoint 和 Run ID。脚本生成四份合成 HDF5 样本，调用
官方 `scripts/official/v1_2/nano.py train_single`，使用单模态、小裁剪、patch size 1，
保留现有模型/Loss/优化器实现；关闭 W&B 和下游评测。去掉 `--mlflow` 可验证禁用路径。

真实数据入口示例（替换数据路径并按资源调整批大小）：

```bash
python scripts/official/v1_2/nano.py train_single nano-mlflow local \
  --common.save_folder=/data/checkpoints/nano-mlflow \
  --dataset.h5py_dir=/data/olmoearth-h5 \
  --trainer.callbacks.wandb.enabled=false \
  --trainer.callbacks.mlflow.enabled=true \
  --trainer.callbacks.mlflow.experiment_name=olmoearth \
  --trainer.callbacks.mlflow.dataset_uri=oss://YOUR_BUCKET/DATASET \
  --trainer.callbacks.mlflow.dataset_version=YOUR_VERSION
```

OSS URI 此处仅为元数据，不增加存储适配器。跨目录恢复使用
`--trainer.load_path=CHECKPOINT_PATH`，保留 Trainer state 加载。旧 Checkpoint
没有 MLflow state 时可指定 `--trainer.callbacks.mlflow.run_id=RUN_ID`。
必须使用相同服务端；ID 或服务端冲突按 failure_policy 处理，并保留原身份。

## 配置与行为

以下字段均位于 `trainer.callbacks.mlflow`：

| 参数 | 默认值 | 用途 |
|---|---|---|
| enabled | false | 仅 Global Rank 0 启用客户端 |
| tracking_uri | http://127.0.0.1:5000 | 环境变量 MLFLOW_TRACKING_URI 优先 |
| experiment_name | olmoearth | 自动创建或复用 Experiment |
| run_name / run_id | null | 名称默认使用训练名称；可指定已有 ID |
| log_interval | 10 | 首步、每 N 步、末步；评测指标不受采样限制 |
| system_metrics | true | MLflow 内置监控，仅 Rank 0 所在主机 |
| system_metrics_interval | 10 秒 | MLflow 采样环境变量可覆盖 |
| failure_policy | warn | 失败后关闭上报；raise 则使训练失败 |
| evaluation_only | 训练 false、评测 true | 关联已有 Run，不结束训练 Run |
| dataset_name/version/uri | 推断/null/h5py_dir | 数据集身份元数据 |
| dataset_num_samples | 已准备数据集长度 | 不额外扫描 |
| dataset_preprocessing | dataset 和 loader 配置 | 可显式覆盖 |

记录现有 Loss、LR、梯度范数、吞吐等数值指标，保留原训练 Step。非法指标名称
字符会规范化并添加哈希；非有限数值省略。系统指标的 Step 是采样序号。
W&B 可独立开关。配置快照、种子、Git SHA、Python/包版本、平台信息均会记录；
恢复保留初始不可变参数，并另存更新后的配置。启动脚本和环境变量全集不会上传。

DatasetInput 的摘要来自元数据，不是数据内容校验和。Checkpoint 只记录小型
URI/Step/指标 JSON，不上传权重；异步保存从 stepN 路径提取 Step，乱序完成
不会使本进程最新路径标签回退。临时禁用跟踪仍保留已加载的 Run ID。

训练内评测保存已有配置和结果；独立评测及 Checkpoint sweep 可用相同开关和
训练 Run ID。Beaker 评测任务需显式传递配置、安装客户端并保证网络连通。
HTTP 默认超时 3 秒、零重试，artifact 超时 5 秒；对应 MLflow 环境变量可覆盖。
请求在 Trainer bookkeeping 中同步执行；失败后本进程停止上报并清理监控。
严格模式的异步错误通过 post_step/close 回传。

## 已完成验证与限制

30 项客户端/Callback 回归测试通过，覆盖身份恢复、脱敏、异步保存、Rank 守卫、
故障策略、真实 CPU/内存监控和 W&B 回调独立性。新文件 Ruff 和锁检查通过。
上游修改文件保留六项既存 SIM102/TRY004 提示。

```bash
python -m pytest tests_mlflow -q
uv lock --check
```

本机仅有 CPU，官方 Nano 在 CUDA 显存统计处失败。ECS 自动脚本尚未在 GPU
机器实跑；真实多进程、GPU 监控、ECS/OSS、下游算法、在线 W&B 和 MLflow 3.x
服务端兼容性仍需验证。未修改模型、Loss、优化器、DataLoader 或 OLMo-core 源码。


## 本轮审查补充

GPU 检查要求同一设备分别存在计算利用率和显存 MB 指标，避免仅有显存利用率时误判。
验证进程的 HTTP 请求统一限制为 2 秒、零重试，artifact 超时 5 秒。
`MLFLOW_VALIDATION_TIMEOUT` 控制每段训练的最长时间，默认 1800 秒；超时只结束
本验证任务创建的进程组（含 Nano 子进程）。GPU/连接预检查失败不会创建结果目录。
代理只支持 HTTP(S) Tracking URI；认证请使用环境变量，URI 不得包含用户名、密码或查询参数。
自签名证书使用 `MLFLOW_TRACKING_SERVER_CERT_PATH`，不自动关闭证书校验。
CPU 回归测试使用一次性本地 HTTP 服务验证中途断连、真实 Callback 降级、
服务端仍可访问、GPU 指标判据及进程树超时清理；不代表 GPU Nano 端到端已经通过。

## 训练时启用和查看 MLflow

已有训练环境安装客户端后，在原训练命令后添加以下参数即可：

```bash
pip install -e '.[mlflow]'
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
# 追加到原有 Nano/其他训练命令：
# --trainer.callbacks.mlflow.enabled=true
# --trainer.callbacks.mlflow.experiment_name=olmoearth
# --trainer.callbacks.mlflow.run_name=nano-test
# --trainer.callbacks.mlflow.log_interval=1
# --trainer.callbacks.wandb.enabled=false
```

MLflow 默认关闭；W&B 开关独立，最后一行仅在需要关闭 W&B 时使用。
启动日志会显示 MLflow Run ID。在浏览器打开服务端地址，进入 Experiments，
选择配置的 Experiment，再打开对应 Run 的 Metrics，查看 Loss、LR、梯度范数、
吞吐和 system/gpu_* 指标。训练指标横轴选择 Step；系统指标按采样序号或时间查看。
参数和配置见 Parameters / Artifacts；Checkpoint JSON 只包含 URI，不包含权重。
选中同一 Experiment 的多个 Run 可比较指标。恢复训练使用相同服务端和原 Checkpoint，
原 Run 内会继续出现后续 Step。

如果浏览器在自己的电脑、服务端在 ECS，在自己的电脑建立隧道：

```bash
ssh -N -L 15000:127.0.0.1:5000 USER@ECS_HOST
```

然后打开 `http://127.0.0.1:15000`。这是浏览器连接，和训练主机使用的
`MLFLOW_TRACKING_URI` 分开设置；不要把自己电脑的 localhost 地址当成 ECS 地址。

2026-11-18 本地验证：`python -m pytest tests_mlflow -q` 共 39 项通过，
包含本轮新增的 9 项 ECS 工具回归案例；新增/修改脚本 Ruff、格式和语法检查通过。
这些是 CPU 上的客户端与工具测试，未在 GPU 上运行完整 Nano 验证脚本。
