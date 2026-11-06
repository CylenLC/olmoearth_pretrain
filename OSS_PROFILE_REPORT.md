# OlmoEarth v1.2 Nano · boto3 Full GET 直读 OSS

2026-11-06，阿里云 ECS 单卡 NVIDIA L20，基线提交 `1a806c2cbcbe6fbaf0f5fcf4820db561284fa5f1`。

## 结论

- **196.8 对 263.7 samples/s 的差距，约 73% 落在训练更新计时区间，约 27% 落在主进程取数等待。** boto3 的 8 次长等合计 37.63 秒。即使把这 8 段全部消掉，按其余时间不变估算，吞吐也只有约 **212.9 samples/s**。
- **较高的 CPU 消耗集中在 DataLoader worker。** 历史 boto3 组比 OSSFS2 多用约 **3.75 个 worker 核**，其中约 **2.12 核是系统态**。短窗直接测到 GET 线程平均每对象耗用 **20.04 ms CPU**，约 **82%** 在读取响应体。短窗 GET 合计约 3.92 核，HDF5 读约 3.53 核，其中 dataset 读取和解码约 2.79 核。
- **8 次长等全部落在 worker 0 的轮转槽位。** 默认按序交付会把这一槽的迟到传播给训练线程。短窗捕获了启动期两种不同的延迟，一次以 worker 内有序 GET 等待为主，一次有 3.93 秒发生在 worker 产出 batch 之后。历史 8 次没有逐 GET 或 batch-ready 事件，无法把它们逐次归到网络、解码或交付。
- **step 200 后的 `Aborted` 在 18-step 短跑中复现。** 两次都发生在最后一个更新完成后的 worker 清理。短跑实测 5 秒排空期限到达后，仍有 4 个 worker 正在 GET、随后继续产出 batch。日志有 5 次 native `SIGABRT`，训练进程最终 exit 0。
- **W8C2 CPU affinity 短窗 A/B 有正向信号，但没有证明 worker CPU 竞争是主要瓶颈。** 一对 24-step 运行中，steps 9–20 吞吐由默认的 169.8 升到隔离的 202.9 samples/s，三个 trace step 的 kernel gap 合计下降 12.8%；但主线程 runqueue wait 不到 1 ms/step、worker CPU 总量近乎不变，而且只有一对运行。没有把 affinity 接入生产路径。

## 1. 审计与复测口径

原 200-step 启动器设置了 `PYTHONPATH=$RUN_DIR/observer:…`，而 `sitecustomize.py` 位于各 run 的共同父目录下。两个输出目录均缺 GET 和 CUDA event 文件。更早的 v1.3 对照 observer 用的是 aiohttp / aiobotocore / s3fs hook，该层也无法覆盖此提交的 boto3 H5 数据路径。当前数据路径由 worker 内的线程池调用 `dataloader._download_oss_payload`，同步 boto3 `get_object` 返回响应头，再读完并关闭 body，最后在 worker 主线程按序把 bytes 交给 HDF5。单独包装 `dataset._download_oss_payload` 会漏掉 DataLoader 已导入的别名。

本次只新增 benchmark 工具。启动器显式执行 observer，worker 从原 initializer 显式进入 observer，同时覆盖上述两个入口。GET 记录请求到头部、HTTP attempt、body 读、完整下载和错误类型。子进程统一用 Linux 单调时钟，并记录线程 CPU 时间。样本侧记录有序 Future 等待、HDF5 读、数据集切片、样本组装、collate 与 worker batch-ready。主进程记录 `next()`、pin、H2D、forward、backward、optimizer、zero-grad 与 shutdown。日志不收集认证头、凭据或异常正文。回环 HTTP 检查走实际线程池和 botocore，验证 503 → 200 的两次 attempt 与 3 字节响应体。关闭 observer 时该检查按预期失败，开启后通过。现有 `test_oss_full_get.py` **4 项通过**。

| 项目 | 设置 |
|---|---|
| 配方 | 官方 v1.2 Nano，双视图，原 AdamW，单 L20 |
| 输入 | 原 3,996 对象前缀，30 次重复，epoch 1，样本 seed 3622、模型 seed 12536 |
| 批处理 | global batch 512，microbatch 64 × 8，W8 × C2，prefetch factor 4，persistent workers |
| 短窗 | 18 个真实 optimizer steps，CPU/CUDA Kineto 只采 steps 9–11 |
| 输入顺序 | 原始与本次的 epoch 索引文件 SHA-256 相同，见下方证据索引 |
| 软件 | Python 3.12.3、PyTorch 2.9.1+cu128、boto3/botocore 1.40.55、h5py 3.15.1、olmo-core 2.6.0 |

保留了模型、数据顺序、训练语义、并发和预取参数，未执行 200-step 重测。短跑复用历史对象池，未清缓存。Kineto 扰动明显，首版在 step 11 后原地停止和导出 trace，耗时 **112.1 s**，worker 在此期间继续预取。因此 steps 12–18 的取数等待只代表已经积累了预取的状态。现交付启动器把 trace 导出移到训练返回后，并提供 `--events-only`。额外的 events-only 短跑在检查到 L20 被另一进程占用时未启动。下面的 CPU/CUDA 数据均来自已完成的 18-step 运行。

## 2. 历史 200-step 复核

窗口从 step 8 完成到 step 200 完成，共 192 × 512 样本。CPU 核数由进程累计 user/system CPU 秒增量除以窗口墙钟估算。GPU 利用率为约 1 Hz 的活跃度采样。

| 指标 | boto3 Full GET | OSSFS2 FUSE |
|---|---:|---:|
| 墙钟 / 吞吐 | 499.421 s / **196.8 samples/s** | 372.731 s / **263.7 samples/s** |
| 主进程 `next()` 均值 / p50 / 最大 | 203.05 / 0.40 / 13,850 ms | 26.77 / 0.30 / 5,073 ms |
| 更新区间均值 | 2.395 s/step | 1.912 s/step |
| 主机 CPU | 75.74% | 50.27% |
| worker user / system / 合计 | 7.10 / 2.72 / **9.82 核** | 5.47 / 0.59 / **6.06 核** |
| 主进程 user + system | 0.98 核 | 1.05 核 |
| GPU 活跃度采样均值 | 32.23% | 42.56% |

126.69 秒墙钟差可拆为更新区间约 **92.67 s**、`next()` 约 **33.85 s**，其余约 0.18 s。更新计时覆盖 H2D、模型和 loss、backward、optimizer 与 zero-grad，也包含主机发射和等待。按 192 个打印值计算线性分位，boto3 的 `next()` p95/p99 为 **144.4 ms / 5.376 s**，与旧汇总的 0.9 ms / 5.257 s 不一致。均值、中位数、8 次阈值计数及最大值可以从原日志复现。见 boto3 重算数据（本地 `benchmarks/oss_profile/evidence/history-boto3.json`） 和 OSSFS2 重算数据（本地 `benchmarks/oss_profile/evidence/history-ossfs2.json`）。

## 3. GET、CPU 与 DataLoader

steps 9–11 取数开始至最后更新完成共 **12.98 s**。后台共完成 2,538 个 Full GET、11.349 GB 响应体，约 **874 MB/s**。这是包括预取在内的完成速率。2,538 次 HTTP attempt 全部为 200，响应体读出的字节数与 Content-Length 一致，无重试。下载区间平均在途 **11.95**、峰值 **16**，包括下载线程的客户端/SDK 前后处理；配置上限为每 worker 2。全 18 步及清理期间记录 25,616 个完整 GET、114.401 GB、零 GET 异常，消费 9,216 个样本。其余包含正常预取、停机排空及每 worker 的两个前看对象。

| 每对象或样本 | p50 | p95 | p99 | 最大 |
|---|---:|---:|---:|---:|
| 完整 GET 墙钟 | 59.49 ms | 95.27 ms | 130.29 ms | 256.65 ms |
| SDK 调用至响应头 | 21.64 ms | 38.91 ms | 65.30 ms | 125.02 ms |
| 读取响应体 | 35.58 ms | 60.65 ms | 81.17 ms | 235.46 ms |
| GET 线程 CPU | 19.42 ms | 25.98 ms | 32.73 ms | 68.28 ms |
| 已下载样本的 HDF5 读 | 19.58 ms | 30.30 ms | 36.65 ms | 50.04 ms |
| HDF5 dataset 索引、读取和解压 | 14.61 ms | 22.95 ms | 28.08 ms | 42.56 ms |
| worker 内按序等待最早的 GET Future | 0.04 ms | 43.07 ms | 77.64 ms | 220.83 ms |

GET 共 **50.85 thread-CPU 秒**，其中 body read **41.81 s**，发送到响应头的 transport attempt **3.52 s**，其余 SDK 调用前后处理 **5.03 s**。body 读取合并了收包及用户态的 TLS、缓冲、校验和分配等潜在工作，当前观测没有 native 栈来区分这些成本。历史的系统态增量与收包和搬运成本相符，也可能包含 worker 的 IPC 和内存相关系统开销。

HDF5 读共 **45.87 thread-CPU 秒**，内含 dataset 读取及解码 **36.24 s**。样本组装另用 **10.29 s**，collate **26.74 s**。按完成事件归窗，整批 collate 可能跨过窗口边界，各项表示并发工作量，不能直接相加成主线程耗时。窗口内 worker 进程采样约 **9.73 核**，接近历史 boto3 的 9.82 核。短窗内 worker 核数和 GET CPU 的量级支持优先查响应体路径，同时保留 HDF5 和 collate 作为独立成本。见 短窗明细（本地 `benchmarks/oss_profile/evidence/profiled.json`）。

### 8 次长尾等待

| Step | 9 | 33 | 49 | 65 | 73 | 161 | 169 | 185 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 主进程等待（s） | 1.997 | 5.257 | 13.850 | 3.046 | 1.548 | 4.122 | 1.228 | 6.580 |

8 次合计 **37.628 s**，占 boto3 全部 `next()` 时间约 **96.5%**。所有 step 满足 `(step−1) % 8 == 0`，连后续两个最长的不足一秒等待也在同槽位（step 193 为 0.9645 s，step 41 为 0.3198 s）。PyTorch 的 `in_order=True` 与轮转派发，以及本次 worker batch-ready 和主进程交付的时间线，共同支持 **worker 0 的有序交付队头阻塞**。worker 内部也按提交顺序取 Future，慢对象可延迟整批。历史原始数据无法确认 worker 0 为何迟到，或等待有多少发生在 worker 已产出之后。

短跑 step 1 的原生 loader `next()` 等 **28.63 s**，其中有 **16.28 s** 与该批次 worker 内已记录的 ≥10 ms 有序 Future 等待重叠，batch-ready 后仍有 0.62 s；这段不含前面的 worker 启动。step 2 等 **7.48 s**，其中 **3.54 s** 在 worker batch-ready 之前、**3.93 s** 在其后，后半段可能涉及序列化、IPC 和 pin。采样时另外记录了 5 次 pin 完成，墙钟中位 **144 ms**、最大 **923 ms**，这些 pin 事件没有与具体 batch 绑定。短跑 steps 9–11 的 `next()` 各为 0.17–0.39 ms，worker 0 的 step 9 batch 已提前 2.17 s 就绪，**没有复现历史稳态 8 次长等**。见 启动期时间线（本地 `benchmarks/oss_profile/evidence/startup.json`）。

## 4. CPU/CUDA 训练更新

Kineto 用 CPU scope 和 CUDA correlation ID 归类 kernel 与拷贝。backward 在 autograd 线程发射的 kernel 按 launch 时间归到对应 backward 区间。去掉 Tensor subclass redispatch 的嵌套标记，每步是 16 次视图 H2D、16 次 forward、8 次 backward。CPU 栏为主机 scope 的墙钟，GPU 栏为对应活动时长的累加，二者可能重叠。GPU 活跃时间另对跨 stream 活动取并集。

| 阶段，steps 9–11 合计 | 主机墙钟 | GPU 活动 |
|---|---:|---:|
| DataLoader `next()` | 0.63 ms | — |
| H2D | 36.86 ms | 36.92 ms HtoD，0.960 GB |
| Forward 路径 | 6.968 s | 1.033 s kernel + 23.05 ms memcpy |
| Backward 路径 | 5.618 s | 1.979 s kernel + 27.06 ms memcpy + 4.05 ms memset |
| Optimizer | 162.42 ms | 6.72 ms kernel + 0.58 ms memcpy |
| Zero-grad | 4.38 ms | 无归属活动 |
| GPU 活跃时间并集 | — | **3.111 s** |

3 个更新的墙钟约 12.97 s，其中 GPU 活动并集约 **24%**。这只描述 profiler 开启时的窗口。Kernel 发射、主机执行和排队间隙占了明显的时间，不能用这三步量化未采样运行的 GPU 空闲比例。全 trace 的 `cudaStreamSynchronize` 有 **14,108 次、130.9 ms** API 墙钟，包含多个算子路径。loss 的 Python 有限值检查没有独立 scope，这个数字不能归给它。H2D 约 **12.3 ms GPU 活动/step**，optimizer 约 **54 ms 主机墙钟/step**，均明显小于 forward/backward。另记录 forward/backward 中约 **44.8 GB DtoD** 累加传输；其 GPU 拷贝时长低，适合结合发射频率做后续定位。各 step 的 patch、空间尺寸及时间步数都有变化，需用同 batch 形状及相同采样窗口来判断历史 **0.483 s/step** 更新差中的 CPU 争抢和模型工作量。见 CUDA 活动汇总（本地 `benchmarks/oss_profile/evidence/trace-summary.json`）。

## 5. step 200 后的 `Aborted`

历史日志在 step 200 指标之后先打印 `terminate called without an active exception`，随后父进程在 `OlmoEarthDataLoader.__iter__ → batches.close() → close() → _shutdown_workers() → worker.join()` 收到 worker `SIGABRT`。该异常在 generator 清理中被忽略。训练完成并返回 0。

短跑最后一步之后，现有 `close()` 禁止继续派发并尝试排空，约 **5.00 s** 后进入 PyTorch shutdown。此时 worker 0、1、6、7 各有 **2 个** GET 在途，随后各自在 shutdown 开始后 **1.76–16.07 s** 还产生 batch-ready。worker 5 在 shutdown 前约 **12 ms** 刚产出 batch。PyTorch 先停止 pin-memory 消费线程，再通知和 join worker。shutdown 调用持续 **25.02 s**，3 个 worker 当时返回 0，另外 5 个的退出码尚未就绪。stderr 记录 **5 次 native abort**，faulthandler 均显示无 Python frame。父进程最终仍返回 0，观测到相关 worker 已离开进程表。

这条时间线支持 **排空超时后，仍在产出/交付 Tensor 的 worker 遇到 native teardown 竞态**，与已有 DataLoader 源码注释一致。完整 GET observer 未记录请求失败，故障边界在训练更新结束后。短跑没有 native backtrace，具体中止点仍需在最小的提前结束复现中抓 C++ 栈。见 关闭阶段数据（本地 `benchmarks/oss_profile/evidence/profiled.json`）。

### 提前结束修复

持久 worker 路径现在在关闭时先停派发、设置本次 worker runtime 的共享停止信号，并保留 PyTorch 结果消费及 pin 线程。worker 在当前 GET 返回后停止补发，取消还没开始的 Future，等待其余在途请求结束并关闭客户端，丢弃未交付的 batch。父进程消费此前已派发的结果和结束消息后才调用 PyTorch shutdown。此前已派发的单个 batch 如果返回异常，排空会继续消费其余结果。300 秒无结果时保留强制清理兜底。停止信号只在 runtime 关闭时设置，正常 epoch 切换继续复用原 worker。

受控回归走公开迭代器、真实 DataLoader worker 和 OSS GET 预取线程池，首批交付后把两个预取 GET 阻塞 5.5 秒。旧代码在两个请求仍阻塞时进入 shutdown。修复版等待两个 GET 完成，关闭期间没有补发，已消费批次计数保持为 1，worker 以 0 退出。见 [回归用例](tests/integration/data/test_dataloader_lifecycle.py)。

又在实际 OSS 上使用 W8C4（每 worker GET 并发 4）、prefetch factor 4、global batch 512、microbatch 64 × 8、原双视图模型及训练入口跑了 **2 steps**。唯一的命令覆盖是 GET 并发 4。第 2 步结束 2.3 ms 后开始关闭，7 个在途 GET 分布在 worker 1、3、6、7。worker 排空期间 GET 数达到 **7,650**，开始数与完成数相等，错误数为 0；最后一条 GET 在 native shutdown 之前结束。worker shutdown 调用在关闭后 **2.744 s** 开始，关闭总耗时 **4.792 s**。8 个 worker 的 exit code 全为 0，父进程 exit 0，进程表中无 worker 残留，日志没有 `SIGABRT` 或 native abort。明细在 W8C4关闭验证（本地 `benchmarks/oss_profile/evidence/close-w8c4.json`） 和 W8C4进程结果（本地 `benchmarks/oss_profile/evidence/exit-w8c4.json`）。

## 6. Full GET 响应体读取对照

本轮使用真实 OSS 同一组 8 个对象，每轮每个对象读取 16 次，8 个进程各使用 2 个 GET 线程。每个方案各两轮，顺序为整读、1 MiB 分块、预分配 `readinto`、预分配 `readinto`、1 MiB 分块、整读。所有 768 次响应的对象 SHA-256 与对应对象一致。每轮有效响应体约 0.533 GiB，整轮墙钟包括 8 个进程的启动，网络计数为整机收包，因此吞吐和网络数字包含启动及其他主机活动。

| 实现 | GET 线程 CPU 秒/GiB | 进程 user / system 秒/GiB | 对象/s | GET p50 / p95 / p99（ms） | 最大单 worker RSS（MiB） |
|---|---:|---:|---:|---:|---:|
| `body.read()` | 5.36–5.46 | 5.62–5.68 / 3.62–3.64 | 31.88–32.30 | 38.9–46.4 / 98.7–125.1 / 141.2–158.2 | 853–857 |
| 1 MiB 分块读至 EOF 后合并 | 5.50–5.70 | 5.78–5.81 / 3.62–3.77 | 33.11–33.72 | 39.5–42.4 / 105.1–110.7 / 129.4–149.9 | 857–858 |
| `readinto` 预分配，EOF 验证后转 bytes | 5.65–5.83 | 5.60–5.77 / 3.90–3.97 | 33.22–33.32 | 37.6–42.9 / 97.2–112.6 / 144.5–148.3 | 860–862 |

GET 线程 CPU 从客户端调用开始计到响应体验证完成，不含 SHA-256；进程 user/system 还包括线程池、客户端初始化和 SHA-256，不能与线程 CPU 相加。每轮整机接收约 598.75–598.78 MB，包含 TCP/TLS 与背景流量。最大 RSS 是单 worker 的生命周期峰值，并非 8 个进程内存之和。两轮的范围用于表示本次波动，样本数不足以给出可信的置信区间。启动开销、缓存和顺序效应影响对象/s，故约 4% 的分块吞吐差异不足以证明训练会受益。相比整读，分块与 `readinto` 的平均 GET 线程 CPU 分别高约 3.5% 和 6.1%，所以保留现有整读实现。完整对象长度校验和 SDK 请求配置保持不变。本轮未接入训练路径，也未进行训练 A/B。详细逐对象记录见 读取对照数据（本地 `benchmarks/oss_profile/evidence/body-ab-final.json`），脚本见 [body_benchmark.py](benchmarks/oss_profile/body_benchmark.py)。

已安装的 botocore `StreamingBody.read()` 调用 urllib3 的响应读取，完成后验证 Content-Length。若服务端响应包含支持的 checksum，`StreamingChecksumBody` 会在读取时更新并在 EOF 验证。urllib3 继续进入 Python HTTP/TLS socket。实验没有关闭 checksum，也没有绕开 botocore 的请求和重试配置。请求失败会使本轮 benchmark 失败，响应级重试次数未单独记录。

`perf stat` 因 ECS 的 `perf_event_paranoid=4` 被拒绝。隔离目录运行的 py-spy 0.4.3 多进程 native 采样器自身发生 SIGSEGV，随后改为单进程、两个 GET 线程采样。100 Hz 采得 3,072 个栈样本，3 次采样错误；其中 526 个栈经过响应体 `_raw_read`，分别有 261 个经过 `SSL_read`、206 个经过 `BIO_read`、57 个经过 `realloc`。这些计数互相重叠，包含 socket 等待及未符号化帧，不能当作 CPU 时间占比。原始栈在 ECS 的 `benchmarks/oss_profile/runs/body-ab/native-single.raw`，汇总见 native 采样摘要（本地 `benchmarks/oss_profile/evidence/native-body-summary.json`）。现有证据指向 TLS/socket 读取与缓冲分配路径值得进一步拆分，但不能排出其中各项的 CPU 成本。

生命周期修复在 Full GET 读取微基准中保持原样。此前实际 OSS W8C2 两步短跑在关闭时有 11 个 GET 在途，8 个 worker 均 exit 0、没有残留或 abort，见 W8C2 关闭验证（本地 `benchmarks/oss_profile/evidence/close-w8c2.json`）；上一轮 W8C4 短跑和受控在途 GET 回归也通过。Full GET 读取替代方案没有显示可靠收益，因此没有修改 GET/DataLoader 热路径。之后单独做的 CPU 调度 A/B 见下一节；没有跑 200-step。

## 7. CPU 调度与亲和性 A/B

### 环境与采集

ECS 有 **16 个逻辑 CPU、8 个物理核**，CPU sibling 分别为 0–1、2–3、…、14–15。cgroup v2 的 session scope、`user-1002.slice` 和 `user.slice` 的 `cpu.max` 都是 `max 100000`，没有 CPU 配额；有效 cpuset 为 0–15。训练源码没有设置 CPU affinity 或 `taskset`，默认进程和线程都允许在 0–15 上调度。设置 `sched_schedstats=0` 时，`/proc/<pid>/task/<tid>/schedstat` 的第二列仍会计数：两个绑在 CPU 0 的忙循环进程在 0.8 秒内分别增加约 0.400、0.405 秒运行队列等待。这一字段可用来量化训练主线程的 runnable 等待；它不包含线程睡眠等待、GPU 等待或 worker 线程的运行队列等待。验证数据见 CPU 调度器检查（本地 `benchmarks/oss_profile/evidence/cpu-scheduler.json`）。

两组都以 **1a806c2c + 已验证的 worker 生命周期修复** 为代码，使用同一个真实 OSS 对象池和完全相同的训练 argv（保存目录除外）：W8C2、prefetch 4、global batch 512、microbatch 64、种子 3622/12536、相同样本顺序。默认组先跑，隔离组后跑。亲和性组把训练 parent 放在 CPU 0–3（两个完整物理核），8 个 worker 放在 CPU 4–15（六个完整物理核）；两组 `OMP_NUM_THREADS=8` 不变。启动前 L20 均为 0% 利用率、0 MiB 显存；训练期间每秒检查 compute PID，没有发现外部 GPU PID。吞吐只取 steps 9–20；steps 21–23 单独采集 CPU/CUDA trace，不计入吞吐。每组最多 24 steps，未进行 200-step 重测。

两组 steps 9–23 每步都读取 **512 个相同索引**；所有索引顺序、patch 和张量形状一致。两组测量索引的 SHA-256 都是 `08dc665ba2c8d55770e1a30b2066e0899d1a70944da57e61b5131d5723552837`。完整逐步核对见 CPU affinity A/B 汇总（本地 `benchmarks/oss_profile/evidence/cpu-affinity-ab.json`）。

| 指标（steps 9–20） | 默认调度 | CPU 亲和性隔离 |
|---|---:|---:|
| 窗口墙钟 / 吞吐 | 36.181 s / **169.8 samples/s** | 30.277 s / **202.9 samples/s** |
| 更新耗时 mean / p50 / p95 | 2761.5 / 2592.9 / 3617.7 ms | 2373.3 / 2210.8 / 3456.3 ms |
| DataLoader `next()` mean / p50 / p95 / max | 249.3 / 0.270 / 1345.1 / 2988.5 ms | 146.7 / 0.178 / 784.6 / 1054.8 ms |
| 主线程运行队列等待 mean / max | 0.799 / 2.550 ms/step | 0.133 / 0.338 ms/step |
| worker user / system CPU | 7.33 / 2.92 核（合计 10.25） | 7.36 / 2.68 核（合计 10.04） |
| 整机 CPU 利用率 | 73.2% | 72.6% |
| GPU 利用率（1 Hz 均值） | 42.1% | 54.9% |

隔离组的吞吐高 **19.5%**，12 个更新耗时全部更低，平均降低 **14.1%**。两窗口更新耗时合计少 **4.66 s**，`next()` 等待合计少 **1.23 s**。主线程运行队列等待总计仅从 **9.58 ms** 降到 **1.60 ms**，不到总窗口差的 0.14%；因此这次吞吐提升不能归因于主线程被 CPU runqueue 长时间阻塞。worker 总 CPU 占用基本相同，约 10 核；整机仍有 CPU 余量。隔离后 parent/worker 不再共享物理核，GPU 利用率上升，与更少的 host→kernel 空档相符。

以下列出吞吐窗口内每一步的更新及取数等待时间（ms）：

| Step | 更新默认 | 更新隔离 | `next()` 默认 | `next()` 隔离 |
|---:|---:|---:|---:|---:|
| 9 | 2654.3 | 2589.7 | 0.3 | 1054.8 |
| 10 | 2864.5 | 2205.5 | 0.2 | 563.6 |
| 11 | 3469.2 | 2005.3 | 0.4 | 0.2 |
| 12 | 2468.5 | 2337.6 | 0.4 | 0.2 |
| 13 | 2531.6 | 2502.3 | 0.3 | 0.1 |
| 14 | 2436.1 | 2216.2 | 0.3 | 0.1 |
| 15 | 3674.2 | 3479.3 | 0.3 | 0.2 |
| 16 | 3571.5 | 3437.5 | 0.2 | 0.1 |
| 17 | 2325.0 | 2033.3 | 0.2 | 0.2 |
| 18 | 2085.8 | 1826.1 | 0.2 | 140.0 |
| 19 | 2731.1 | 1998.4 | 2988.5 | 0.2 |
| 20 | 2326.2 | 1848.4 | 0.2 | 0.3 |

Kineto 对应的 CUDA kernel 活动如下。`kernel gap` 是该 optimizer step 最早与最晚 kernel 之间的空档总量；不包括首个 kernel 之前和最后一个 kernel 之后的时间。

| Trace step | 默认 kernel active / gap / span | 隔离 kernel active / gap / span |
|---|---:|---:|
| 21 | 1935.9 / 2076.3 / 4012.2 ms | 1935.9 / 1416.2 / 3352.0 ms |
| 22 | 236.4 / 3092.5 / 3328.9 ms | 236.6 / 3002.4 / 3238.9 ms |
| 23 | 1271.6 / 2740.0 / 4011.5 ms | 1269.2 / 2479.7 / 3749.0 ms |
| 合计 | 3443.9 / 7908.8 / 11352.7 ms | 3441.7 / 6898.3 / 10340.0 ms |

CUDA kernel 活动总量基本不变，三个采样 step 的 kernel gap 合计降低 **12.8%**；这支持 affinity 布局改善了 GPU 工作供给，但 trace 只覆盖三个 step。DataLoader 尾延迟仍有波动：默认组 step 19 等待 2.99 s，隔离组 step 9/10 分别等待 1.05/0.56 s。它们落在不同 step，表明 OSS/worker 的短窗抖动仍影响吞吐。GPU 利用率为 1 Hz 采样，短暂外部任务可能漏采；所有采样点的 compute PID 列表都没有外部 PID。

**判断：短窗结果显示 CPU 核心布局可能显著影响 GPU 供给，但尚不足以断定 worker CPU 竞争是主要瓶颈。** affinity A/B 的方向一致（吞吐 +19.5%、kernel gap −12.8%、GPU 利用率 +12.8 个百分点），但只完成一对、默认调度总是先跑；窗口只有 12 个不同形状的更新，而且 `next()` 尾延迟在两组落于不同 step，无法估计运行间误差。主线程 runqueue 等待极少，worker CPU 总量也未明显下降，所以证据不支持“CPU 被用满导致主线程排队”这个解释；可能的核共享、缓存或 worker 供给效应仍需复测确认。**没有把 CPU affinity 接入生产 DataLoader，也没有更改模型、GET、并发、worker 数或 prefetch。**下一步最有价值的是反序再做一对相同短窗 A/B，并记录 worker 主线程及 GET 线程的 schedstat，以判断正向差异能否复现。

## 8. 按 task ID 关联有序交付

在 benchmark observer 中，将 PyTorch 的派发 task ID 贯穿 worker 取任务、GET/Future、HDF5、collate、结果入队、pin 线程取件、pin 完成和主线程交付。GET 在提交时记录 `(worker, request_id)`，Future 消费时使用相同 ID，因此跨 batch 的预取也能关联。worker `Queue.put()` 返回只代表放入本进程发送缓冲，后续到 pin 线程收到之间包含序列化、传输及 pin 线程处理其他 batch 的等待，不应单独称为纯 IPC。主线程交付 ID 的计算针对本配方的 `in_order=True`。

CPU 上的真实 DataLoader 小样本验证了派发至交付的 task ID、线程池 GET/Future 的 request ID，以及 pin 队列前后 task ID；样本输出与未启用 observer 的输出一致。真实 OSS W8C2、prefetch 4、batch 512、microbatch 64 的 24-step 诊断中，24 个交付 task 的这些阶段均有记录，每个已消费 Future 都能找到对应 GET。报告的数据按单调时钟关联；详细结果保留在本地未提交的 `benchmarks/oss_profile/evidence/ordered-task-summary.json`。

**这次没有复现 worker 0 的稳态长尾。** worker 0 的 step 9 和 17 的 `next()` 分别为 0.31 和 0.44 ms，其 batch 分别提前约 7.09 和 10.73 秒完成 worker 侧处理。启动期 step 1 等待 23.97 s，其 worker fetch 为 23.85 s，其中 Future 等待累计 12.62 s、HDF5 读取使用 7.85 thread-CPU 秒；并发墙钟和 CPU 时间不能直接相加。历史 200-step 的 8 次长等没有 task ID，不能从这次追踪反推它们的具体阶段。

本次抓到另一个可定位的尾巴：step 10 / task 9（worker 1）`next()` 等待 **996.6 ms**，目标 batch 在开始等待前 **1291.5 ms** 已经完成 worker fetch。目标结果入队后 **2129.7 ms** 才被 pin 线程取到；期间 pin 线程先处理后续 task 15，其 pin 操作持续 1169.1 ms，与 step 10 的等待重叠 **422.7 ms**。目标 task 9 在等待开始后 838.5 ms 被 pin 线程接收，自己的 pin 操作耗时 132.1 ms，在等待开始后 996.2 ms 输出。这一实例表明后续 batch 的 pin 工作可延迟当前按序消费的 batch，但它属于 worker 1，并不足以证明历史 worker 0 的 8 次长等由同一机制造成。

GPU 启动前检查为 0 MiB、无 compute PID；运行后有其他实验的 compute PID 出现，在 164 个 GPU 样本中占 113 个。因此以下数值只描述受干扰的诊断运行，**不能用于性能 A/B 或优化收益判断**：steps 9–24 为 171.3 samples/s，`next()` 均值 / p50 / 最大为 89.36 / 0.322 / 996.6 ms，worker CPU user / system 为 6.75 / 2.32 核，GPU 利用率 1 Hz 均值为 40.7%（包含外部进程）。24 步完成后有 9 个 GET 在途，16970 个 GET 开始和结束数相等、无错误；8 个 worker exit 0、无残留和 abort。

在用户确认 GPU 空闲后，又连续完成两次相同配置、相同 observer 的 24-step 短跑。两轮启动前 GPU 均空闲，全部 1 Hz GPU 采样没有外部 compute PID；steps 9–24 的每个 task 都是同样 512 个样本索引，顺序与输入形状逐步一致，索引 SHA-256 均为 `985e3a6b8ff0b1e4178e25549bfc292453e604c5d3d1f2206c069fbef57b8b10`。这些都是优化前诊断，并非前后对照。

| 指标，steps 9–24 | 干净短跑 1 | 干净短跑 2 |
|---|---:|---:|
| 吞吐 | 193.4 samples/s | 183.2 samples/s |
| `next()` mean / p50 / max | 0.374 / 0.347 / 0.665 ms | 0.405 / 0.325 / 1.098 ms |
| worker 0 的 step 9 / 17 `next()` | 0.373 / 0.241 ms | 1.098 / 0.470 ms |
| worker CPU user / system | 7.29 / 2.79 核 | 7.61 / 2.89 核 |
| GPU 利用率，1 Hz 均值 | 53.25% | 43.50% |
| 关闭时在途 GET / 8 个 worker 退出码 | 14 / 全为 0 | 11 / 全为 0 |

两轮 24 个 task 的阶段关联完整，所有 Future 均找到对应 GET；关闭时各自所有 GET 都结束、无错误、无 worker 残留或 abort。启动期 step 2 分别等待 11.21 / 13.71 秒；batch 完成 worker fetch 后仍有 3.86 / 4.52 秒，pin 线程在等待窗口内也处理了若干后续 task。这再次验证启动期 pin 工作和队列延迟，但稳态 worker 0 的秒级等待仍未复现。两次相同输入的吞吐相差约 5.3%，不能将单次短窗差异当作优化收益。

本轮没有把性能优化接入默认 DataLoader：目标 worker 0 的稳态尾巴在两次干净短窗中均未复现，无法验证 pin 优先级调整是否解决历史问题，也没有可信的优化后吞吐。下一步应继续收集有 task ID 的短窗直到复现目标尾巴；若目标 batch 确认被后续 pin 阻塞，再评估保持有序消费和关闭语义的最小候选方案，并使用同样观测开销进行前后比较。

## 9. 优先级

| 优先级 | 后续改动方向 | 证据与验收 |
|---|---|---|
| P0 | 提前结束的 worker 生命周期 | **已修复并验收。** 阻塞 GET 的控制回归在旧实现上于五秒处失败，在新实现上于 5.5 秒释放后通过。实际 OSS W8C4 两步短跑在 7 个 GET 在途时关闭，所有请求完成，8 个 worker 都以 exit 0 退出且没有残留。 |
| P1 | 降低 Full GET 响应体每字节 CPU | body 占 GET 线程 CPU 的约 82%，历史增量集中在 worker，系统态占增量约 56%。用相同 Full GET 语义拆出 socket/TLS、缓冲、checksum、复制和分配成本。先比较 CPU 秒/GiB、完整性、吞吐和 p99，保留长度校验与对象顺序。 |
| P1 | 缩短 worker 0 批次的有序交付尾延迟 | 历史八次长等全在同槽位。下一次短窗优先记录 worker batch-ready、发送、pin 和有序交付的同一 batch ID，检验瓶颈在 Future、HDF5、collate 还是 IPC。保持交付顺序，先定位再优化。 |
| P2 | 复核 parent/worker 核心隔离 | 单对 A/B 中 throughput +19.5%、kernel gap −12.8%，但主线程 runqueue wait 很低、worker CPU 总量相近。反序重复相同短窗并采 worker schedstat 后，再决定是否保留生产 affinity。 |
| P2 | 给 HDF5/组装及训练 host 发射做同形状对照 | HDF5 读在短窗约 45.87 CPU 秒，历史更新区间解释约 73% 的墙钟差。对同一 batch 形状比较 boto3 与 OSSFS2 的 worker CPU、CUDA 活跃并集和 host 发射间隙，判断先做数据热路径还是训练算子。 |
| P3 | H2D 和 optimizer 放在后面 | H2D 约 12.3 ms GPU 活动/step，optimizer 约 54 ms 主机墙钟/step。当前证据不足以把它们作为 0.483 s/step 差距的主因。 |

## 证据与复跑

本地 evidence 目录包含运行生成的摘要和采样数据，作为未提交的实验产物保留；报告中的数字提供可提交的结果摘要。

- 历史两组重算、启动期、CPU/CUDA 采样及退出摘要（本地 `benchmarks/oss_profile/evidence`）
- CPU 调度器/cgroup 检查（本地 `benchmarks/oss_profile/evidence/cpu-scheduler.json`）、CPU affinity A/B 对照及样本顺序校验（本地 `benchmarks/oss_profile/evidence/cpu-affinity-ab.json`）、默认组 events 摘要（本地 `benchmarks/oss_profile/evidence/cpu-default-short.json`）、隔离组 events 摘要（本地 `benchmarks/oss_profile/evidence/cpu-isolated-short.json`）、默认/隔离 CUDA trace 汇总（本地 `benchmarks/oss_profile/evidence/cpu-default-trace.json`） / 隔离组（本地 `benchmarks/oss_profile/evidence/cpu-isolated-trace.json`）
- [显式启动器](benchmarks/oss_profile/launch.py)、[observer](benchmarks/oss_profile/observer.py)、[回环验证](benchmarks/oss_profile/check_observer.py)、[分析器](benchmarks/oss_profile/analyze.py)
- ECS 历史只读证据在 `/tmp/olmoearth-nano-storage-ab-20261106TEL5zhg/{boto3-oss,mounted-ossfs2-configured}/`，短窗原始 GET/worker/parent JSONL、训练日志及 Kineto gzip trace 在 `/tmp/olmoearth-oss-profile.9CU7rW/olmoearth_pretrain/benchmarks/oss_profile/runs/boto3/`。
- 本次原始 CPU A/B 日志、1 Hz GPU 采样及 Kineto traces 在 ECS 隔离副本的 `benchmarks/oss_profile/runs/cpu-default-20261107/` 和 `benchmarks/oss_profile/runs/cpu-isolated-20261107/`；本地 evidence 目录保存了可复核的 JSON 汇总。两组都完成关闭验证：GET 开始/结束数相同、无 GET 错误、8 个 worker exit 0、无残留进程或 abort。
- task ID 诊断的原始日志在 ECS 隔离副本的 `benchmarks/oss_profile/runs/ordered-task-observe-20261107/`；对应分析器为 `benchmarks/oss_profile/ordered_analysis.py`。本次结果包含外部 GPU 活动，不是有效的性能对照。
- 两轮干净诊断的原始日志在同一隔离副本的 `benchmarks/oss_profile/runs/ordered-task-clean2/` 和 `ordered-task-clean3/`；本地未提交的 `benchmarks/oss_profile/evidence/ordered-clean2-*.json` 和 `ordered-clean3-*.json` 含逐 task 摘要、窗口指标和关闭校验。
- epoch 索引 SHA-256 为 `f2f45ca21b6546692cffeac40d2493fb3edf695d174260ac074e5b0b4f6465f6`，原始 trace SHA-256 为 `b25943d774310cdc1b63fe13c3dd5a95ae64ba5b0985d221b5a6f7a3f0600190`。

在 ECS 的隔离 worktree 内，通过已有的私有凭据包装器启动。启动器把步数限制在 24 以内，每次要求新的输出目录。

```sh
py=/home/qupengfei/code/OlmoEarth/.venv/bin/python
baseline=/tmp/olmoearth-nano-storage-ab-20261106TEL5zhg/boto3-oss/command.json
"$py" benchmarks/oss_profile/launch.py "$baseline" benchmarks/oss_profile/runs/replay-w8c4 \
  --steps 2 --first 1 --last 1 --events-only --verify-clean-close --gets-per-worker 4
```
