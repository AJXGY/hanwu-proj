# 0324proj 使用指南

## 项目概述

这是一个基于 PyTorch 的推理延迟估测工具，支持单卡、单机多卡 TP、跨 host 多卡 TP 的推理性能预测与实测对比。

## 环境

- **Python**: `~/miniconda3/envs/llama_4gpu/bin/python` (3.10.12, torch 2.6.0+cu126)
- **启动器**: `python -m torch.distributed.run --standalone`（系统中无 `torchrun` 命令）
- **GPU**: 4x A100 80GB PCIe (GPU 0-3)
- **模型路径**:
  - 1B: `/home/o_zhanghui/projs/0320proj/downloads/Llama-3.2-1B`
  - 8B: `/home/o_zhanghui/projs/0320proj/downloads/Meta-Llama-3.1-8B`

## Dashboard 运行约定

- dashboard 支持两种 runner：`docker_run_image` 与 `local_python`
- `local_python` 必须固定使用独立虚拟环境：`/home/o_zhanghui/miniconda3/envs/llama_4gpu/bin/python`
- `local_python` 不依赖当前 shell 的 `conda activate` / `PATH`
- `local_python` 会设置 `PYTHONNOUSERSITE=1`，避免混入 `~/.local/lib/python*/site-packages`
- `local_python` 会为每次任务创建隔离的本机 MPS 目录并拉起独立 MPS daemon，不复用宿主机其他用户已有的 MPS pipe
- 单机 host 运行时，`--physical-devices` 使用宿主机**全局 GPU 编号**，例如 `2,3`
- 单机 host 运行时，不要再额外手工设置 `CUDA_VISIBLE_DEVICES`；代码会直接按 `physical_devices` 选择物理卡
- 推荐边界：单机场景优先 `local_python`；跨 host 场景优先两边都用 Docker
- 不建议让本机 dashboard 通过 SSH 远程拉起对面机器的 host venv；跨 host 更适合在对面机器本地起 Docker rank

## 核心入口

- 主脚本：`torch_infer_mvp.py`
- 参数定义：`mvp_execution.py:15`
- 输出报告：`<output-dir>/report.json`

## 文件职责速查

| 文件 | 职责 |
|---|---|
| `torch_infer_mvp.py` | 入口，仅 `from mvp_app import main; main()` |
| `mvp_app.py` | 主流程：标定 → 导出图 → 估测 → TP切分 → 实测 → 对比 → 写报告 |
| `mvp_execution.py` | 参数解析 + `ExecutionConfig` 构建 + `dist.init_process_group` |
| `mvp_parallel.py` | `apply_tensor_parallel()`：对每层 Linear 做 Col/Row-wise TP |
| `mvp_runtime.py` | `extract_inference_graphs()` 导出 prefill/decode 图；`prepare_runtime_inputs()` |
| `mvp_calibration.py` | 硬件标定：GEMM TFLOPS、attention TFLOPS、带宽、launch overhead |
| `mvp_estimator.py` | 逐节点估测、module substitution、通信预测、phase summary |
| `mvp_graph.py` | TP shard-aware 缩放（`tp_shard_node_estimate`）、shape/target 归一化 |
| `mvp_measurement.py` | `distributed_cuda_wall_time_ms()`：TP 下以 `max(rank_times)` 为 phase latency |
| `mvp_profile.py` | `profile_cuda_ops()`：TorchDispatchMode + torch.profiler 抓 op/collective |
| `mvp_types.py` | 所有 dataclass：`ExecutionConfig`、`NodeEstimate`、`PhaseSummary` 等 |

## 三种运行模式

### 1. 单卡模式

```bash
~/miniconda3/envs/llama_4gpu/bin/python torch_infer_mvp.py \
  --model-path /home/o_zhanghui/projs/0320proj/downloads/Llama-3.2-1B \
  --prompt "your prompt" \
  --dtype bf16 \
  --device cuda:0 \
  --parallel-mode single \
  --physical-devices 0 \
  --world-size 1 \
  --tp-size 1 \
  --warmup 0 \
  --benchmark-repeat 1 \
  --profile-repeat 1 \
  --max-new-tokens 8 \
  --output-dir /tmp/output
```

### 2. 单机双卡 TP

```bash
~/miniconda3/envs/llama_4gpu/bin/python -m torch.distributed.run --standalone --nproc_per_node 2 torch_infer_mvp.py \
  --model-path /home/o_zhanghui/projs/0320proj/downloads/Llama-3.2-1B \
  --prompt "your prompt" \
  --dtype bf16 \
  --device cuda:0 \
  --parallel-mode tp \
  --physical-devices 0,1 \
  --world-size 2 \
  --tp-size 2 \
  --warmup 0 \
  --benchmark-repeat 1 \
  --profile-repeat 1 \
  --max-new-tokens 8 \
  --output-dir /tmp/output
```

### 3. 双机双卡 TP（每机 1 卡）

**Host 0（主节点）：**
```bash
torchrun \
  --nnodes 2 \
  --nproc_per_node 1 \
  --node_rank 0 \
  --master_addr <host0_ip> \
  --master_port 29500 \
  torch_infer_mvp.py \
    --model-path /path/to/model \
    --prompt "your prompt" \
    --dtype bf16 \
    --device cuda:0 \
    --parallel-mode tp \
    --physical-devices 0 \
    --world-size 2 \
    --tp-size 2 \
    --nnodes 2 \
    --nproc-per-node 1 \
    --node-rank 0 \
    --master-addr <host0_ip> \
    --master-port 29500 \
    --interconnect ethernet \
    --dist-timeout-minutes 15 \
    --warmup 0 \
    --benchmark-repeat 1 \
    --profile-repeat 1 \
    --max-new-tokens 8 \
    --output-dir /tmp/output
```

**Host 1（从节点）：**
```bash
torchrun \
  --nnodes 2 \
  --nproc_per_node 1 \
  --node_rank 1 \
  --master_addr <host0_ip> \
  --master_port 29500 \
  torch_infer_mvp.py \
    --model-path /path/to/model \
    --prompt "your prompt" \
    --dtype bf16 \
    --device cuda:0 \
    --parallel-mode tp \
    --physical-devices 0 \
    --world-size 2 \
    --tp-size 2 \
    --nnodes 2 \
    --nproc-per-node 1 \
    --node-rank 1 \
    --master-addr <host0_ip> \
    --master-port 29500 \
    --interconnect ethernet \
    --dist-timeout-minutes 15 \
    --warmup 0 \
    --benchmark-repeat 1 \
    --profile-repeat 1 \
    --max-new-tokens 8 \
    --output-dir /tmp/output
```

## TP=4 推理（单机四卡）

```bash
~/miniconda3/envs/llama_4gpu/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 torch_infer_mvp.py \
  --model-path /home/o_zhanghui/projs/0320proj/downloads/Llama-3.2-1B \
  --prompt "your prompt" \
  --max-new-tokens 2 \
  --dtype bf16 \
  --parallel-mode tp \
  --physical-devices 0,1,2,3 \
  --world-size 4 \
  --tp-size 4 \
  --device cuda:0 \
  --warmup 1 \
  --benchmark-repeat 1 \
  --profile-repeat 1 \
  --output-dir validation_reports/tp4_smoke
```

## 关键参数约束

- `--parallel-mode tp` 时：`len(physical_devices) == tp_size == world_size`
- host `local_python` 运行时：`--physical-devices` 是**全局 GPU 索引**（例如 `2,3`）
- Docker 运行时：`--physical-devices` 仍按**容器视角**填写
- `--device` 始终传 `cuda:0`（local_rank 由 torchrun 注入）
- host `local_python` 路径不要手工加 `CUDA_VISIBLE_DEVICES`

## TP 切分细节

- 仅支持 Llama 风格的 `model.model.layers` 结构
- 每层内部做 TP：
  - **Colwise**: `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`
  - **Rowwise**: `o_proj`, `down_proj`
- `num_heads` 和 `num_key_value_heads` 除以 `tp_size`
- 注意：**图导出在 TP 切分之前**，所以估测时需要手动做 shard-aware 缩放

## 预测侧流程（在 mvp_app.py）

1. `build_calibration()` → 跑 microbenchmark 得到硬件参数
2. `extract_inference_graphs()` → 用 `torch.export.export()` 导出 prefill/decode 图
3. 遍历图节点做 `estimate_node()` → 按 op family 估 FLOPs/bytes/time
4. `apply_tensor_parallel()` → 原地切分模型
5. `tp_shard_node_estimate()` → 把单卡 estimate 按 1/tp_size 缩放
6. `build_predicted_comm()` → 启发式预测 all_reduce 通信时间
7. `summarize_phase_with_module_substitution()` → 用实测 module 时间覆盖对应 node
8. 得到 prefill / decode_step / request 预测

## 测量侧流程

1. `distributed_cuda_wall_time_ms()` 测 prefill_fn / decode_fn / request_fn
2. TP 下每轮前 `dist.barrier()`，phase latency = `max(rank_times)`
3. `profile_cuda_ops()` 额外 profile 一次，抓 op 序列和 CUDA 事件
4. `build_operator_compare_rows()` 做 predicted vs measured 对齐
5. 识别 collective（all_reduce / all_gather / reduce_scatter）

## 报告输出

- 输出目录：`validation_reports/<run_name>/`
- 文件：`report.json`（完整数据）、`report.md`（可读版）、`dashboard_status.json`
- 核心指标在 `comparison` 字段：
  - `prefill_relative_error_pct`
  - `decode_step_relative_error_pct`
  - `request_relative_error_pct`
- `operator_compare.summary.coverage_estimate_ms_pct` → 估测覆盖率
- `rank_measurements` → 每 rank 独立 timing
- `comm` → 实测 + 预测的通信时间

## 跨 Host 估测关键要点

### 1. 双机启动顺序
- **必须先启动从节点（rank1），再启动主节点（rank0）**
- 从节点会等待主节点建立连接
- 如果主节点先启动，会超时失败

### 2. 网络配置
- 确保两台机器之间网络可达
- 设置 `NCCL_SOCKET_IFNAME` 指定通信网卡
- 常见网卡名：`ens1f0`、`eth0`、`ib0`

### 3. 模型路径
- 两台机器的模型路径可以不同
- 但模型内容必须一致
- 推荐使用相同版本的模型文件

### 4. 报告输出
- 只有 rank0 会输出完整报告
- 从节点不输出 `report.json`
- 查看结果只需看主节点的输出目录

## 误差调优经验

### 1. 当前精度水平（Llama-3.1-8B 测试）
- `prefill` 误差：约 1-5%
- `decode_step` 误差：约 5-10%
- `request` 误差：约 1-10%

### 2. 影响精度的关键因素
- **Profile 粒度**：跨 host decode 必须用 `decoder_layer` 粒度，不能用 `self_attn/mlp`
- **计时方式**：跨 host decode 必须用 `wall_time`，不能用 `cuda_event`
- **Request 组合**：必须用 `profiled decode loop average`，不能用单步 decode 乘步数

### 3. 调优策略
- 如果 `decode_step` 误差大，检查是否用了正确的 profile 粒度
- 如果 `request` 误差大，检查是否用了正确的 decode 组合策略
- 跨 host 场景下，通信模型精度也会影响结果

## Docker 运行注意事项

### 1. GPU 设备映射
- Docker 的 `--gpus device=X` 是物理 GPU 编号
- 容器内的 `cuda:0` 是容器视角的第一张卡
- 多卡时注意映射关系

### 2. 网络模式
- 跨 host 必须用 `--network host`
- 单机 TP 可以用默认网络模式

### 3. 内存配置
- 大模型建议加 `-e PYTORCH_ALLOC_CONF=expandable_segments:True`
- 避免内存碎片化问题

## 常见问题排查

### 1. NCCL 超时
- 检查网络连通性：`ping <host_ip>`
- 检查防火墙设置
- 增加 `--dist-timeout-minutes` 参数

### 2. 模型加载失败
- 确认模型路径正确
- 确认模型文件格式完整
- 检查磁盘空间

### 3. GPU 内存不足
- 减少 `--max-new-tokens`
- 使用更小的模型
- 检查是否有其他进程占用 GPU

## 实战技巧（基于近期复现）

### 1. 先用稳定采样参数再看误差
- 推荐默认用 `--warmup 2 --benchmark-repeat 5 --profile-repeat 10` 再评估误差；`--warmup 0 --benchmark-repeat 1 --profile-repeat 1` 波动会明显更大。
- `warmup` 是预热轮，不计入 `samples_ms`；`benchmark_repeat` 是正式实测采样轮数。
- `profile_repeat` 仅用于 module profiling 采样，和实测采样解耦。
- TP/分布式下每轮 phase latency 取各 rank 耗时最大值（`max(rank_times)`），所以小样本更容易受偶发慢 rank 影响。

### 2. Docker 的 `--gpus` 传参要区分“本地执行”与“SSH 嵌套执行”
- 本地直接执行（无额外 shell 嵌套）通常用：`--gpus device=0,1`。
- 通过 `ssh '...docker run ...'` 远端执行时，建议写成：`--gpus '\"device=0,1\"'`。
- 如果遇到报错 `cannot set both Count and DeviceIDs on device request`，优先检查这里的引号层级。

### 3. 跨 host 复现顺序与最小检查清单
- 启动顺序：先远端 `rank1`，再本地 `rank0`。
- 先同步代码再启动：`rsync -az --delete --exclude ".git" --exclude "__pycache__" ...`。
- 启动前至少确认三件事：两边模型路径存在、两边 Docker 镜像存在、`NCCL_SOCKET_IFNAME` 是可通信网卡。
- 结果只看 `rank0` 输出目录（跨 host 场景从节点不写主报告）。

### 4. `physical-devices` 的语义
- `local_python`：`--physical-devices` 按宿主机全局 GPU 编号填写，例如 `2,3`。
- `docker_run_image`：`--physical-devices` 按容器视角填写。
- 例如宿主机 `--gpus device=2,3`，容器内通常映射为 `cuda:0,cuda:1`，此时 Docker 参数应传 `--physical-devices 0,1`。

### 5. 关于“异常值筛除”要点
- `sanitize_module_profiles()` 只作用于估测侧的 module profile 替换数据。
- 不会删除 `measured.prefill/decode_step/request.samples_ms` 中的实测样本。
- 当前 outlier 回退策略是把异常 module profile 回退到 covered estimate，用于避免估测侧被异常 profile 拉偏。

### 6. 已验证的参考组合（仅供复现参考）
- 单机 4 卡 TP=4（1B，`warmup=2/benchmark_repeat=5/profile_repeat=10`）：`prefill` 误差约 `2.94%`，`decode_step` 约 `13.35%`，`request` 约 `4.41%`。
- 双机各 1 卡 TP=2（8B，`warmup=2/benchmark_repeat=5/profile_repeat=10`）：`prefill` 约 `4.52%`，`decode_step` 约 `12.36%`，`request` 约 `0.30%`。
- 双机各 2 卡 TP=4（8B，`warmup=2/benchmark_repeat=5/profile_repeat=10`）：`prefill` 约 `4.74%`，`decode_step` 约 `12.73%`，`request` 约 `0.44%`。
- 单机 4 卡 TP=4 快速冒烟（1B，`warmup=0/benchmark_repeat=1/profile_repeat=1`）：`prefill` 约 `2.73%`，`decode_step` 约 `46.23%`，`request` 约 `7.34%`（小样本下 decode 波动显著）。
- 双机各 1 卡 TP=2 快速冒烟（8B，`warmup=0/benchmark_repeat=1/profile_repeat=1`）：`prefill` 约 `3.43%`，`decode_step` 约 `7.88%`，`request` 约 `0.09%`。
- 上述数字依赖当时机器负载、网卡状态和模型路径，作为量级参考，不作为硬阈值。

### 7. 跨 host “看起来卡住”时的首查项
- 现象：`rank0` 命令长时间不退出，像是卡死。
- 首先检查远端 `rank1` 容器是否已退出：`ssh <host1> "docker ps -a | rg cross-host-r1"`。
- 重点看远端日志：`ssh <host1> "docker logs cross-host-r1"`；最常见根因是代码版本不一致（例如远端仍是 `--repeat`，本地已切到 `--benchmark-repeat/--profile-repeat`）。
- 建议固定流程：跨机每次开跑前都执行一次 `rsync -az --delete --exclude ".git" --exclude "__pycache__" ...`，再启动 rank1/rank0。

### 8. 运行环境一致性建议
- 若目标是稳定复现，优先用 Docker（当前常用镜像：`huggingface/transformers-pytorch-gpu:latest`）。
- 跨 host 场景优先保持两边都是 Docker，不要混成“一边 host venv、一边 Docker”，否则环境漂移更难排查。
- dashboard 的 `local_python` 已处理 `~/.local/lib/python*/site-packages` 污染问题，但前提是使用固定解释器 `/home/o_zhanghui/miniconda3/envs/llama_4gpu/bin/python`。
- 若手工本地跑，先确认 `python -c "import transformers; print(transformers.__file__)"` 指向期望环境，再执行完整流程。
- 若手工本地跑遇到 `Error 805: MPS client failed to connect`，优先检查是否为当前任务单独创建了自己的 `CUDA_MPS_PIPE_DIRECTORY`。

### 9. 跨 host 最小健康检查（开跑前 30 秒）
- 本机到远端路由：`ip route get <host1_ip>`，远端到本机同理；确认走的是预期网卡（如 `ens1f0` / `ens1f0np0`）。
- 两端镜像与模型目录存在：`docker images | rg transformers-pytorch-gpu`、`ls <model_path>`。
- 端口唯一：为本次任务指定未占用的 `master_port`（避免与历史任务冲突）。

### 10. 跨 host 最小测试流程（推荐先跑 smoke test）
- 目标：先验证双机互联、分布式初始化与主报告输出，再扩大到正式参数。
- 推荐 smoke 参数：`--warmup 0 --benchmark-repeat 1 --profile-repeat 1 --max-new-tokens 1`
- 推荐先用较小模型，例如 `/home/o_zhanghui/projs/0320proj/downloads/Llama-3.2-1B`
- 两边保持完全一致：代码版本一致、镜像一致、模型版本一致、`master_addr/master_port/world_size/tp_size/nnodes` 一致
- 启动顺序固定：先远端 `rank1`，后本地 `rank0`

**远端 `rank1` 最小 smoke 模板：**
```bash
docker run --rm --network host --gpus '"device=0"' \
  -v /path/to/0324proj:/workspace \
  -v /path/to/model_dir:/path/to/model_dir:ro \
  -w /workspace \
  ubuntu2204-torch26-py:latest \
  bash -lc "python3 -m torch.distributed.run \
    --nnodes 2 \
    --nproc_per_node 1 \
    --node_rank 1 \
    --master_addr <host0_ip> \
    --master_port 29500 \
    torch_infer_mvp.py \
      --model-path /path/to/model_dir/Llama-3.2-1B \
      --prompt 'alpha alpha alpha alpha alpha alpha alpha alpha' \
      --dtype bf16 \
      --device cuda:0 \
      --parallel-mode tp \
      --physical-devices 0 \
      --world-size 2 \
      --tp-size 2 \
      --nnodes 2 \
      --nproc-per-node 1 \
      --node-rank 1 \
      --master-addr <host0_ip> \
      --master-port 29500 \
      --interconnect ethernet \
      --dist-timeout-minutes 15 \
      --warmup 0 \
      --benchmark-repeat 1 \
      --profile-repeat 1 \
      --max-new-tokens 1 \
      --output-dir /tmp/cross_host_smoke_rank1"
```

**本地 `rank0` 最小 smoke 模板：**
```bash
docker run --rm --network host --gpus device=0 \
  -v /home/o_zhanghui/projs/0324proj:/workspace \
  -v /home/o_zhanghui/projs/0320proj/downloads:/home/o_zhanghui/projs/0320proj/downloads:ro \
  -w /workspace \
  ubuntu2204-torch26-py:latest \
  bash -lc "python3 -m torch.distributed.run \
    --nnodes 2 \
    --nproc_per_node 1 \
    --node_rank 0 \
    --master_addr <host0_ip> \
    --master_port 29500 \
    torch_infer_mvp.py \
      --model-path /home/o_zhanghui/projs/0320proj/downloads/Llama-3.2-1B \
      --prompt 'alpha alpha alpha alpha alpha alpha alpha alpha' \
      --dtype bf16 \
      --device cuda:0 \
      --parallel-mode tp \
      --physical-devices 0 \
      --world-size 2 \
      --tp-size 2 \
      --nnodes 2 \
      --nproc-per-node 1 \
      --node-rank 0 \
      --master-addr <host0_ip> \
      --master-port 29500 \
      --interconnect ethernet \
      --dist-timeout-minutes 15 \
      --warmup 0 \
      --benchmark-repeat 1 \
      --profile-repeat 1 \
      --max-new-tokens 1 \
      --output-dir /tmp/cross_host_smoke_rank0"
```

**成功判据：**
- `rank0` 正常退出并写出 `/tmp/cross_host_smoke_rank0/report.json`
- `report.json` 中 `execution.parallel_mode == "tp"`
- `report.json` 中 `execution.placements` 同时包含 `rank0` 与 `rank1`
- 从节点无需写主报告；只要不异常退出即可

**失败时优先检查顺序：**
- 先看远端 `rank1` 容器日志，再看本地 `rank0` 日志
- 检查两边代码是否同步，尤其参数名是否一致（`benchmark_repeat/profile_repeat` 等）
- 检查 `master_addr/master_port/NCCL_SOCKET_IFNAME` 是否一致且可达
- 检查两边 `--gpus` 与 `--physical-devices` 是否匹配各自容器视角

## 工具脚本

### 0. 从历史 report 构建 module profile 打表库
```bash
~/miniconda3/envs/llama_4gpu/bin/python tools/build_module_profile_table.py \
  --reports-glob "validation_reports/**/report.json" \
  --table-db-path database/module_profile_table.jsonl
```

### 1. 跨 host all-reduce 基准测试
```bash
torchrun \
  --nnodes 2 \
  --nproc_per_node 1 \
  --node_rank 0 \
  --master_addr <host0_ip> \
  --master_port 29500 \
  tools/bench_allreduce.py
```

### 2. 快速误差检查
```bash
~/miniconda3/envs/llama_4gpu/bin/python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path('/tmp/output/report.json').read_text())
for phase, est, measured in [
    ('prefill', report['estimate']['prefill']['end_to_end_time_ms'], report['measured']['prefill']['mean_ms']),
    ('decode_step', report['estimate']['decode_step']['end_to_end_time_ms'], report['measured']['decode_step']['mean_ms']),
    ('request', report['estimate']['request_end_to_end_time_ms'], report['measured']['request']['mean_ms']),
]:
    rel = abs(est - measured) / measured * 100 if measured else 0.0
    print(phase, f'est={est:.4f}', f'measured={measured:.4f}', f'rel_err={rel:.4f}%')
PY
```

## 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-path` | 模型路径 | `/opt/models/Llama-3.2-1B` |
| `--prompt` | 输入 prompt | 默认 prompt |
| `--max-new-tokens` | 生成 token 数 | 4 |
| `--dtype` | 数据类型 | `bf16` |
| `--parallel-mode` | 并行模式 | `single` |
| `--physical-devices` | 物理设备编号 | 自动检测 |
| `--world-size` | 总进程数 | 1 |
| `--tp-size` | TP 大小 | 1 |
| `--nnodes` | 节点数 | 1 |
| `--nproc-per-node` | 每节点进程数 | 1 |
| `--node-rank` | 当前节点排名 | 0 |
| `--master-addr` | 主节点地址 | `127.0.0.1` |
| `--master-port` | 主节点端口 | 29500 |
| `--interconnect` | 互联类型 | `auto` |
| `--dist-timeout-minutes` | 分布式超时 | 30 |
| `--estimate-only` | 仅估测模式 | False |
| `--estimate-mode` | 模块 profile 来源：`online` / `table` / `hybrid` | `online` |
| `--table-db-path` | 打表数据库路径（jsonl） | `database/module_profile_table.jsonl` |
| `--table-writeback` | 将在线采集的 module profile 回写到打表库 | False |
| `--warmup` | 预热次数 | 2 |
| `--benchmark-repeat` | 实测重复次数 | 5 |
| `--profile-repeat` | profiling 重复次数 | 10 |
| `--output-dir` | 输出目录 | `reports/torch_mvp` |

## 已知限制

- TP 仅支持 Llama 风格模型（依赖 `model.model.layers`）
- 图导出在 TP 之前，通信预测是启发式模型（只看 row-wise GEMM → all_reduce）
- TP 模式要求 `world_size == tp_size`（当前不支持 `pp * tp` 等混合并行）
