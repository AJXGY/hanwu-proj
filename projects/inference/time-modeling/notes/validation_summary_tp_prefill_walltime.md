# 当前代码实验总结：TP Prefill Wall-Time

## 说明

这份 note 只保留**当前代码实现下重新跑出来的最新结果**。

旧的实验结果、旧的中间结论、以及之前为了定位问题而跑出的诊断性数据，已经不再作为本 note 的主结论依据。

当前结论对应的代码分支：`fix/single-gpu-phase-overestimate`

当前关键提交：

- `922465d`：接入 `online / table / hybrid`，补齐打表读写与导表链路，并修复 decode benchmark 复用 cache 导致的测量不稳定问题
- `3b2fc1a`：将 TP prefill 的 profiling 改为 `wall_time`，同时保留 `self_attn/mlp` 粒度
- `56c8801`：统一 `request` 指标定义，移除双机场景下“半估测、半实测”的 request 计算方式
- `d2f52bb`：记录过一版单卡 residual launch overhead 降权 checkpoint，但它会回归 TP2/TP4，因此不是最终方案
- `4391c19`：将单卡 module profiling scope 收敛到 `layer_plus_tail`，并修复 layer scope 覆盖匹配的前缀误命中

## 这份 note 里的结果来源

这份 note 里的每个拓扑都包含两种模式：

- `online`
- `table`

其中：

1. `online`：运行时直接在线采集 `module_profile`
2. `table`：先用该拓扑刚跑出来的 `online/report.json` 构建一份新的 table，再立刻用这份新 table 重新跑 `table`

因此，这里的 `table` 结果不是历史旧表，而是**当前环境下 freshly built 的表**。

## 指标定义

当前 `request` 已统一为纯估测口径：

`request_est = prefill_est + (max_new_tokens - 1) * decode_step_est`

这意味着：

- `prefill`
- `decode_step`
- `request`

现在都属于同一套纯估测语义。

双机场景里虽然仍然保留 `request_decode_profile_steps_ms` 作为诊断信息，但它**不会再参与** `request_end_to_end_time_ms` 的计算。

## 共享实验配置

除非单独说明，下面所有结果都使用：

- 模型：`Meta-Llama-3.1-8B`
- prompt：`alpha alpha alpha alpha alpha alpha alpha alpha`
- `max_new_tokens=8`
- `warmup=1`
- `benchmark-repeat=3`
- `profile-repeat=3`
- Docker 镜像：`huggingface/transformers-pytorch-gpu:latest`
- 入口脚本：`torch_infer_mvp.py`

上一轮矩阵实验输出目录：`/tmp/mvp_matrix_current`

上一轮单卡诊断与单机 TP 复核输出目录：`/tmp/mvp_module_instrumentation`

本轮 phase adjustment 复核输出目录：`/tmp/mvp_phase_adjustment_check`

## 如何运行

### 总体原则

每个拓扑都分两步：

1. 跑 `online`
2. 用 `online/report.json` 生成 table，再跑 `table`

table 的生成命令统一是：

```bash
docker run --rm \
  -e PYTHONPATH=/workspace \
  -v /home/o_zhanghui/projs/0324proj:/workspace \
  -v /tmp:/tmp \
  -w / \
  --entrypoint python3 \
  huggingface/transformers-pytorch-gpu:latest \
  /workspace/tools/build_module_profile_table.py \
    --reports-glob "tmp/<ONLINE_REPORT_PATH>/report.json" \
    --table-db-path /tmp/<TABLE_DB_PATH>.jsonl \
    --overwrite
```

### 1. 单机单卡

#### online

```bash
docker run --rm \
  --gpus all \
  --ipc host \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e CUDA_VISIBLE_DEVICES=3 \
  -v /home/o_zhanghui/projs/0324proj:/workspace \
  -v /home/o_zhanghui/projs/0320proj/downloads/Meta-Llama-3.1-8B:/model:ro \
  -v /tmp/mvp_matrix_current/single/online:/output \
  --entrypoint python3 \
  huggingface/transformers-pytorch-gpu:latest \
  /workspace/torch_infer_mvp.py \
    --model-path /model \
    --prompt "alpha alpha alpha alpha alpha alpha alpha alpha" \
    --dtype bf16 \
    --device cuda:0 \
    --parallel-mode single \
    --physical-devices 0 \
    --world-size 1 \
    --tp-size 1 \
    --warmup 1 \
    --benchmark-repeat 3 \
    --profile-repeat 3 \
    --max-new-tokens 8 \
    --estimate-mode online \
    --output-dir /output
```

#### table

```bash
docker run --rm \
  --gpus all \
  --ipc host \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e CUDA_VISIBLE_DEVICES=3 \
  -v /home/o_zhanghui/projs/0324proj:/workspace \
  -v /home/o_zhanghui/projs/0320proj/downloads/Meta-Llama-3.1-8B:/model:ro \
  -v /tmp/mvp_matrix_current/single/table:/output \
  -v /tmp:/hosttmp \
  --entrypoint python3 \
  huggingface/transformers-pytorch-gpu:latest \
  /workspace/torch_infer_mvp.py \
    --model-path /model \
    --prompt "alpha alpha alpha alpha alpha alpha alpha alpha" \
    --dtype bf16 \
    --device cuda:0 \
    --parallel-mode single \
    --physical-devices 0 \
    --world-size 1 \
    --tp-size 1 \
    --warmup 1 \
    --benchmark-repeat 3 \
    --profile-repeat 3 \
    --max-new-tokens 8 \
    --estimate-mode table \
    --table-db-path /hosttmp/mvp_matrix_current/single/table_db.jsonl \
    --output-dir /output
```

### 2. 单机双卡

使用本机空闲的两张卡。当前实验使用的是 `CUDA_VISIBLE_DEVICES=0,1`。

#### online

```bash
docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -v /home/o_zhanghui/projs/0324proj:/workspace \
  -v /home/o_zhanghui/projs/0320proj/downloads/Meta-Llama-3.1-8B:/model:ro \
  -v /tmp/mvp_matrix_current/tp2/online:/output \
  --entrypoint torchrun \
  huggingface/transformers-pytorch-gpu:latest \
  --standalone \
  --nproc_per_node 2 \
  /workspace/torch_infer_mvp.py \
    --model-path /model \
    --prompt "alpha alpha alpha alpha alpha alpha alpha alpha" \
    --dtype bf16 \
    --device cuda:0 \
    --parallel-mode tp \
    --physical-devices 0,1 \
    --world-size 2 \
    --tp-size 2 \
    --nnodes 1 \
    --nproc-per-node 2 \
    --node-rank 0 \
    --warmup 1 \
    --benchmark-repeat 3 \
    --profile-repeat 3 \
    --max-new-tokens 8 \
    --estimate-mode online \
    --output-dir /output
```

#### table

把上面的 `--estimate-mode online` 改成：

```bash
--estimate-mode table \
--table-db-path /hosttmp/mvp_matrix_current/tp2/table_db.jsonl
```

并挂载：

```bash
-v /tmp:/hosttmp
```

### 3. 单机四卡

当前实验使用本机 `CUDA_VISIBLE_DEVICES=0,1,2,3`。

#### online

```bash
docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -v /home/o_zhanghui/projs/0324proj:/workspace \
  -v /home/o_zhanghui/projs/0320proj/downloads/Meta-Llama-3.1-8B:/model:ro \
  -v /tmp/mvp_matrix_current/tp4/online:/output \
  --entrypoint torchrun \
  huggingface/transformers-pytorch-gpu:latest \
  --standalone \
  --nproc_per_node 4 \
  /workspace/torch_infer_mvp.py \
    --model-path /model \
    --prompt "alpha alpha alpha alpha alpha alpha alpha alpha" \
    --dtype bf16 \
    --device cuda:0 \
    --parallel-mode tp \
    --physical-devices 0,1,2,3 \
    --world-size 4 \
    --tp-size 4 \
    --nnodes 1 \
    --nproc-per-node 4 \
    --node-rank 0 \
    --warmup 1 \
    --benchmark-repeat 3 \
    --profile-repeat 3 \
    --max-new-tokens 8 \
    --estimate-mode online \
    --output-dir /output
```

#### table

同样把 `--estimate-mode online` 改成：

```bash
--estimate-mode table \
--table-db-path /hosttmp/mvp_matrix_current/tp4/table_db.jsonl
```

### 4. 双机各一卡

本机：`ICT107`，IP `10.208.130.107`

远端：`jumpserver-nvidia-185`，实际主机 `ICT185`

当前实验使用：

- 本机：`CUDA_VISIBLE_DEVICES=3`
- 远端：`CUDA_VISIBLE_DEVICES=1`
- 本机网卡：`ens1f0`
- 远端网卡：`ens1f0np0`

#### online

先起远端 rank1：

```bash
ssh jumpserver-nvidia-185 '
docker run -d \
  --name matrix-1x1-online-r1 \
  --gpus all \
  --network host \
  --ipc host \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e NCCL_SOCKET_IFNAME=ens1f0np0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v /home/o_zhanghui/projs/0324proj:/workspace \
  -v /home/o_zhanghui/projs/0320proj/Meta-Llama-3.1-8B-ms:/model:ro \
  -v /tmp/mvp_matrix_current/2host_1x1/online_remote:/output \
  --entrypoint torchrun \
  huggingface/transformers-pytorch-gpu:latest \
  --nnodes 2 \
  --nproc_per_node 1 \
  --node_rank 1 \
  --master_addr 10.208.130.107 \
  --master_port 29554 \
  /workspace/torch_infer_mvp.py \
    --model-path /model \
    --prompt "alpha alpha alpha alpha alpha alpha alpha alpha" \
    --dtype bf16 \
    --device cuda:0 \
    --parallel-mode tp \
    --physical-devices 0 \
    --world-size 2 \
    --tp-size 2 \
    --nnodes 2 \
    --nproc-per-node 1 \
    --node-rank 1 \
    --master-addr 10.208.130.107 \
    --master-port 29554 \
    --interconnect ethernet \
    --dist-timeout-minutes 15 \
    --warmup 1 \
    --benchmark-repeat 3 \
    --profile-repeat 3 \
    --max-new-tokens 8 \
    --estimate-mode online \
    --output-dir /output'
```

再起本机 rank0：

```bash
docker run --rm \
  --name matrix-1x1-online-r0 \
  --gpus all \
  --network host \
  --ipc host \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e NCCL_SOCKET_IFNAME=ens1f0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v /home/o_zhanghui/projs/0324proj:/workspace \
  -v /home/o_zhanghui/projs/0320proj/downloads/Meta-Llama-3.1-8B:/model:ro \
  -v /tmp/mvp_matrix_current/2host_1x1/online:/output \
  --entrypoint torchrun \
  huggingface/transformers-pytorch-gpu:latest \
  --nnodes 2 \
  --nproc_per_node 1 \
  --node_rank 0 \
  --master_addr 10.208.130.107 \
  --master_port 29554 \
  /workspace/torch_infer_mvp.py \
    --model-path /model \
    --prompt "alpha alpha alpha alpha alpha alpha alpha alpha" \
    --dtype bf16 \
    --device cuda:0 \
    --parallel-mode tp \
    --physical-devices 0 \
    --world-size 2 \
    --tp-size 2 \
    --nnodes 2 \
    --nproc-per-node 1 \
    --node-rank 0 \
    --master-addr 10.208.130.107 \
    --master-port 29554 \
    --interconnect ethernet \
    --dist-timeout-minutes 15 \
    --warmup 1 \
    --benchmark-repeat 3 \
    --profile-repeat 3 \
    --max-new-tokens 8 \
    --estimate-mode online \
    --output-dir /output
```

#### table

流程：

1. 用本机 rank0 产出的 `online/report.json` 导表
2. 把 `table_db.jsonl` 用 `rsync` 拷到远端相同 `/tmp` 路径
3. 双边都把 `--estimate-mode online` 改成：

```bash
--estimate-mode table \
--table-db-path /hosttmp/mvp_matrix_current/2host_1x1/table_db.jsonl
```

### 5. 双机各两卡

当前实验使用：

- 本机：`CUDA_VISIBLE_DEVICES=0,1`
- 远端：`CUDA_VISIBLE_DEVICES=0,1`
- 本机网卡：`ens1f0`
- 远端网卡：`ens1f0np0`

#### online

远端先起：

```bash
ssh jumpserver-nvidia-185 '
docker run -d \
  --name matrix-2x2-online-r1 \
  --gpus all \
  --network host \
  --ipc host \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e NCCL_SOCKET_IFNAME=ens1f0np0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v /home/o_zhanghui/projs/0324proj:/workspace \
  -v /home/o_zhanghui/projs/0320proj/Meta-Llama-3.1-8B-ms:/model:ro \
  -v /tmp/mvp_matrix_current/2host_2x2/online_remote:/output \
  --entrypoint torchrun \
  huggingface/transformers-pytorch-gpu:latest \
  --nnodes 2 \
  --nproc_per_node 2 \
  --node_rank 1 \
  --master_addr 10.208.130.107 \
  --master_port 29574 \
  /workspace/torch_infer_mvp.py \
    --model-path /model \
    --prompt "alpha alpha alpha alpha alpha alpha alpha alpha" \
    --dtype bf16 \
    --device cuda:0 \
    --parallel-mode tp \
    --physical-devices 0,1 \
    --world-size 4 \
    --tp-size 4 \
    --nnodes 2 \
    --nproc-per-node 2 \
    --node-rank 1 \
    --master-addr 10.208.130.107 \
    --master-port 29574 \
    --interconnect ethernet \
    --dist-timeout-minutes 15 \
    --warmup 1 \
    --benchmark-repeat 3 \
    --profile-repeat 3 \
    --max-new-tokens 8 \
    --estimate-mode online \
    --output-dir /output'
```

本机再起：

```bash
docker run --rm \
  --name matrix-2x2-online-r0 \
  --gpus all \
  --network host \
  --ipc host \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e NCCL_SOCKET_IFNAME=ens1f0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v /home/o_zhanghui/projs/0324proj:/workspace \
  -v /home/o_zhanghui/projs/0320proj/downloads/Meta-Llama-3.1-8B:/model:ro \
  -v /tmp/mvp_matrix_current/2host_2x2/online:/output \
  --entrypoint torchrun \
  huggingface/transformers-pytorch-gpu:latest \
  --nnodes 2 \
  --nproc_per_node 2 \
  --node_rank 0 \
  --master_addr 10.208.130.107 \
  --master_port 29574 \
  /workspace/torch_infer_mvp.py \
    --model-path /model \
    --prompt "alpha alpha alpha alpha alpha alpha alpha alpha" \
    --dtype bf16 \
    --device cuda:0 \
    --parallel-mode tp \
    --physical-devices 0,1 \
    --world-size 4 \
    --tp-size 4 \
    --nnodes 2 \
    --nproc-per-node 2 \
    --node-rank 0 \
    --master-addr 10.208.130.107 \
    --master-port 29574 \
    --interconnect ethernet \
    --dist-timeout-minutes 15 \
    --warmup 1 \
    --benchmark-repeat 3 \
    --profile-repeat 3 \
    --max-new-tokens 8 \
    --estimate-mode online \
    --output-dir /output
```

#### table

流程与 `2host 1x1` 相同：

1. 本机 rank0 的 `online/report.json` 导表
2. `rsync` 到远端 `/tmp/mvp_matrix_current/2host_2x2/table_db.jsonl`
3. 双边都改成 `--estimate-mode table`

## 最新结果总表

说明：本轮再次刷新了 `单机单卡 / 单机双卡 / 单机四卡` 三档结果；两档跨机结果仍沿用上一轮矩阵数据。

| 拓扑 | 模式 | Prefill 误差 | Decode 误差 | Request 误差 |
|---|---|---:|---:|---:|
| 单机单卡 | online | 0.22% | 3.29% | 0.67% |
| 单机单卡 | table | 3.65% | 3.83% | 3.69% |
| 单机双卡 | online | 2.51% | 1.42% | 2.12% |
| 单机双卡 | table | 1.77% | 1.44% | 1.71% |
| 单机四卡 | online | 2.67% | 2.72% | 2.88% |
| 单机四卡 | table | 0.70% | 0.81% | 1.86% |
| 双机各一卡 | online | 13.87% | 15.36% | 17.03% |
| 双机各一卡 | table | 13.04% | 15.87% | 11.36% |
| 双机各两卡 | online | 3.75% | 16.42% | 17.71% |
| 双机各两卡 | table | 10.81% | 19.83% | 17.73% |

## 本轮新增结论

1. 单卡高估的主因已经可以进一步确定为 phase 级的系统偏差，而不只是 module scope 过细。
2. 参考 TP 的 `compute + comm` 结构后，单卡引入了一条独立的 `phase_adjustment` 通道，显式吸收 `module_sum` 与真实 phase wall time 之间的差值。
3. 这条做法对单卡非常有效：`online request` 误差从上一轮的 `9.02%` 进一步降到 `0.67%`，`table request` 从 `13.45%` 降到 `3.69%`。
4. 这次改动没有回归单机 TP：`tp2/tp4` 的 `online/table` 仍保持在 `1%~3%` 左右。
5. 当前单卡不再明显弱于单机 TP，本地拓扑下的主要问题已经基本转移到跨机场景。

## 本轮 phase adjustment 结论

新增机制：单卡 `phase_adjustment`

语义：

- `phase_adjustment = measured_phase_wall_time - base_phase_estimate`
- 其中 `base_phase_estimate = module_sum + residual_runtime_overhead`

对应关系上，它相当于把单卡也改成了类似 TP 的两段式结构：

- TP：`module compute + comm`
- 单卡：`module compute + phase_adjustment`

本轮单卡在线采集到的校正项为：

- `prefill phase_adjustment_time_ms = -1.6253`
- `decode_step phase_adjustment_time_ms = -1.8703`

含义：当前单卡即便已经把 scope 收敛到 `layer_plus_tail`，`module_sum` 仍会略高于真实 phase wall time，因此需要一条负的 phase-level correction，而不是继续把误差硬塞回 per-node launch overhead。

## 本轮单卡诊断

新增脚本：`tools/diagnose_single_module_profile_overhead.py`

核心诊断输出：`/tmp/mvp_module_instrumentation/single_module_profile_diag.json`

真实单卡 phase wall time：

- `prefill`：`20.1398` ms
- `decode_step`：`18.9703` ms

模块求和与真实 phase 的比值：

- `submodule + cuda_event`
  - `prefill`：`1.1661x`
  - `decode_step`：`1.1405x`
- `submodule + wall_time`
  - `prefill`：`1.3220x`
  - `decode_step`：`1.3370x`
- `layer + cuda_event`
  - `prefill`：`1.0264x`
  - `decode_step`：`1.0116x`
- `layer_plus_tail + cuda_event`
  - `prefill`：`1.0466x`
  - `decode_step`：`1.0487x`

补充观察：把 `submodule + cuda_event` 按 chunk 分批采集，只能把比值从大约 `1.14x` 降到 `1.13x` 左右，说明主问题不是“同时挂了太多 hook”，而是单卡下 `submodule` profiling boundary 本身太细。

因此，本轮代码改动采用：

1. 单卡 `prefill/decode_step` 都切到 `layer_plus_tail`
2. 保留 TP 路径原有 scope 选择不变
3. 修复 layer scope 覆盖匹配中的前缀误命中问题，避免 `model.layers.1` 错配到 `model.layers.10/11/...`

## 逐项结果说明

### 1. 单机单卡

#### online

- `prefill`：estimate `19.8302` ms，measured `19.7877` ms，误差 `0.2153%`
- `decode_step`：estimate `19.2549` ms，measured `18.6423` ms，误差 `3.2850%`
- `request`：estimate `150.2295` ms，measured `151.2501` ms，误差 `0.6748%`

#### table

- `prefill`：estimate `19.8302` ms，measured `19.1830` ms，误差 `3.6483%`
- `decode_step`：estimate `19.2549` ms，measured `17.9792` ms，误差 `3.8307%`
- `request`：estimate `150.5584` ms，measured `145.1997` ms，误差 `3.6906%`

说明：

- 单卡误差相比上一轮再次明显下降，尤其 `online request` 已经压到了 `1%` 以内。
- `table` 也同步显著改善，说明这次引入的不是一次性的在线测量技巧，而是可导表、可复用的 phase-level 建模项。
- 当前单卡 `decode_step` 仍略高于 `prefill`，但已经进入和单机 TP 同一量级。

### 2. 单机双卡

#### online

- `prefill`：estimate `133.8607` ms，measured `137.3114` ms，误差 `2.5133%`
- `decode_step`：estimate `126.1705` ms，measured `124.4118` ms，误差 `1.4159%`
- `request`：estimate `1017.0541` ms，measured `1039.0890` ms，误差 `2.1211%`

#### table

- `prefill`：estimate `133.8590` ms，measured `136.2728` ms，误差 `1.7676%`
- `decode_step`：estimate `126.1677` ms，measured `124.3802` ms，误差 `1.4403%`
- `request`：estimate `1017.0327` ms，measured `1034.6828` ms，误差 `1.7062%`

说明：

- 这一档依然很稳，`online/table` 都在 `1%~3%`。
- `table` 仍然完整命中：`prefill 64/64`，`decode 64/64`。
- 首次重跑时出现过一次 GPU0 空闲显存不足导致的 OOM；同命令重跑后成功，属于环境问题，不是代码问题。

### 3. 单机四卡

#### online

- `prefill`：estimate `132.0452` ms，measured `135.6702` ms，误差 `2.6689%`
- `decode_step`：estimate `124.4541` ms，measured `127.9351` ms，误差 `2.7200%`
- `request`：estimate `1003.2238` ms，measured `1032.9942` ms，误差 `2.8825%`

#### table

- `prefill`：estimate `132.0548` ms，measured `132.9760` ms，误差 `0.6989%`
- `decode_step`：estimate `124.4690` ms，measured `125.4788` ms，误差 `0.8066%`
- `request`：estimate `1003.3377` ms，measured `1022.3693` ms，误差 `1.8622%`

说明：

- 单机四卡仍然非常准。
- 这次 `table` 仍然比 `online` 更稳一些，说明本轮单卡 phase adjustment 也没有破坏 TP4 的打表路径。

### 4. 双机各一卡

#### online

- `prefill`：estimate `148.4966` ms，measured `130.4109` ms，误差 `13.8683%`
- `decode_step`：estimate `148.3520` ms，measured `128.5992` ms，误差 `15.3599%`
- `request`：estimate `1186.9603` ms，measured `1014.1963` ms，误差 `17.0346%`

#### table

- `prefill`：estimate `148.0763` ms，measured `130.9928` ms，误差 `13.0416%`
- `decode_step`：estimate `148.2152` ms，measured `127.9183` ms，误差 `15.8670%`
- `request`：estimate `1185.5826` ms，measured `1064.6050` ms，误差 `11.3636%`

说明：

- 一旦跨 host，误差就明显变大。
- `prefill` 和 `decode` 大约都在 `13%` 到 `16%` 左右。
- `table` 和 `online` 在 phase 级别上基本同一量级。
- 这组 `table` 仍然完整命中：`prefill 64/64`，`decode 32/32`。
- 这里 decode 是 `decoder_layer` 粒度，不是 `self_attn/mlp`。

### 5. 双机各两卡

#### online

- `prefill`：estimate `155.4111` ms，measured `161.4618` ms，误差 `3.7474%`
- `decode_step`：estimate `153.4937` ms，measured `131.8467` ms，误差 `16.4183%`
- `request`：estimate `1229.8669` ms，measured `1044.7893` ms，误差 `17.7143%`

#### table

- `prefill`：estimate `155.4775` ms，measured `140.3151` ms，误差 `10.8059%`
- `decode_step`：estimate `153.5144` ms，measured `128.1138` ms，误差 `19.8265%`
- `request`：estimate `1230.0780` ms，measured `1044.7991` ms，误差 `17.7334%`

说明：

- 这是当前最难的一档。
- `decode` 和 `request` 误差都比较大。
- `table` 没有把问题明显变好，和 `online` 基本在同一量级。
- 这里同样是完整命中：`prefill 64/64`，`decode 32/32`。

## 当前结论

1. 本地拓扑里，单卡、单机双卡、单机四卡都已经进入可用区间。
2. 单卡通过 `phase_adjustment` 后已经不再明显弱于单机 TP，尤其 `online` 精度已经非常高。
3. 单机双卡、单机四卡仍然很稳，当前误差基本保持在 `1%~3%`。
4. 当前单卡高估的主要来源，可以概括为 `module_sum` 与真实 phase wall time 之间的系统偏差；这类偏差更适合用 phase-level correction 建模，而不是继续压进 residual launch overhead。
5. 跨机场景仍然是当前主要短板。
6. 双机各一卡与双机各两卡中，`table` 没有明显比 `online` 更差，但也没有根本解决跨机误差。

## 剩余问题

1. 单卡 `phase_adjustment` 是否能稳定泛化到不同 prompt length、不同 `max_new_tokens`、以及不同 GPU 型号。
2. 双机场景为什么 decode 仍然稳定高估，尤其 `2host 2x2` 更明显。
3. 能否把当前单卡 `phase_adjustment` 的思路进一步推广到跨机 phase 建模，而不只是局限在本地单卡。
4. 为什么跨机场景下 `table` 和 `online` 在 phase 级别表现非常接近，说明当前主要瓶颈不在 table 命中，而在跨机 phase 建模本身。

## 本轮实验结果文件绝对路径

- 单机单卡
  - `/tmp/mvp_phase_adjustment_check/single/online/report.json`
  - `/tmp/mvp_phase_adjustment_check/single/table/report.json`
  - `/tmp/mvp_phase_adjustment_check/single/table_db.jsonl`
  - `/tmp/mvp_module_instrumentation/single_module_profile_diag.json`
- 单机双卡
  - `/tmp/mvp_phase_adjustment_check/tp2/online/report.json`
  - `/tmp/mvp_phase_adjustment_check/tp2/table/report.json`
  - `/tmp/mvp_phase_adjustment_check/tp2/table_db.jsonl`
- 单机四卡
  - `/tmp/mvp_phase_adjustment_check/tp4/online/report.json`
  - `/tmp/mvp_phase_adjustment_check/tp4/table/report.json`
  - `/tmp/mvp_phase_adjustment_check/tp4/table_db.jsonl`
- 双机各一卡
  - `/tmp/mvp_matrix_current/2host_1x1/online/report.json`
  - `/tmp/mvp_matrix_current/2host_1x1/table/report.json`
- 双机各两卡
  - `/tmp/mvp_matrix_current/2host_2x2/online/report.json`
  - `/tmp/mvp_matrix_current/2host_2x2/table/report.json`
