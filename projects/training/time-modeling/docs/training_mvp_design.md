# 训练 MVP 设计

## 1. 目标

本工作区的训练逻辑改为参考 `train-infer-estimation` 的训练估算方法：

- 以训练 iteration 为主指标。
- 保留预测值、实测值和相对误差。
- 训练并行主线使用 Tensor Parallel。
- 不再把 Pipeline Parallel 作为默认训练路线。

当前优先覆盖：

- Llama 架构。
- 单机寒武纪 MLU。
- `parallel_mode=single/tp`。
- `tp_size=1/2`。
- 单次训练 iteration 时间。
- 误差目标 `< 20%`。

## 2. 时间语义

对外主指标：

```text
train_iteration_time
  = sum(forward_time + backward_compute_time + backward_comm_time for each microbatch)
  + optimizer_step_time
```

其中：

- `forward_time`: 一个 microbatch 的前向时间。
- `backward_compute_time`: 一个 microbatch 的反向计算时间。
- `backward_comm_time`: TP 场景下反向通信时间；单卡为 0。
- `optimizer_step_time`: 所有 microbatch 完成后的参数更新时间。

当前 report 使用：

```text
T_real = measured.train_iteration_time_ms
T_sim  = estimate.train_iteration_time_ms
error  = comparison.train_iteration_relative_error_pct
```

## 3. TP 实现策略

TP 训练入口是：

```text
torch_train_tp_mvp.py
```

它使用 PyTorch 原生 tensor parallel API：

- `torch.distributed.run` 启动多进程。
- `torch.distributed.device_mesh.init_device_mesh` 构造 TP mesh。
- `parallelize_module` 对 Llama layer 做张量并行。
- `ColwiseParallel` 应用于 `q_proj/k_proj/v_proj/gate_proj/up_proj`。
- `RowwiseParallel` 应用于 `o_proj/down_proj`。

设备后端：

- CUDA 使用 `nccl`。
- MLU 使用 `cncl`。

## 4. 训练负载

当前训练负载是 `lora_vocab_adapter`，已按 `train-infer-estimation` 的训练任务对齐：

- backbone 不冻结。
- forward 不使用 `no_grad`。
- hidden state 不 `detach`。
- autograd 穿过完整 Llama backbone。
- adapter 结构为 `hidden_size -> rank -> vocab_size`。
- loss 使用 token-level vocab cross entropy。
- optimizer 只更新 adapter 参数。
- 使用 synthetic microbatch。
- 默认优化器是 `sgd`。
- 默认开启 `gradient checkpointing`。

因此当前任务不是 pooled hidden state 后的 2 类分类，也不是冻结 backbone 的轻量训练。
在 `Llama-3.1-8B`、`lora_rank=8` 下，adapter 参数量为 `1,058,816`。

## 5. 静态估算与 profile

默认 `T_sim` 已切到 `train-infer-estimation` 风格的静态 estimator：

- forward: 使用模型结构 FLOPs、TP effective scale、`tp_forward_efficiency` 校准。
- backward_compute: 使用训练模式对应的 backward 计算模型和校准项。
- backward_comm: 使用 ring all-reduce latency/bandwidth/overlap 公式。
- optimizer_step: 使用 optimizer memory bytes、带宽和效率校准。

训练组件 profile 单独记录，并对齐 `train-infer-estimation` 的四阶段语义：

- `forward`
- `backward_compute`
- `backward_comm`
- `optimizer_step`

组合规则：

```text
T_sim = microbatch_count * (forward + backward_compute + backward_comm) + optimizer_step
```

profile 支持两种模式，但只作为 `profile_estimate` 对照，不再是默认 `T_sim` 来源：

- `online`: 当前运行现场测量并可写入 JSONL。
- `table`: 只从 JSONL 表读取已有 profile。

默认 profile 表：

```text
database/train_component_profile_cambricon_mlu580_tp_single.jsonl
database/train_component_profile_cambricon_mlu580_tp2.jsonl
```

训练校准配置入口：

```text
configs/train_config.yaml
```

## 6. 报告结构

报告文件：

```text
report.json
report.md
```

关键字段：

- `runtime_model = torch_tp_train_v1`
- `execution.parallel_mode`
- `execution.tp_size`
- `execution.world_size`
- `measured.train_iteration_time_ms`
- `estimate.train_iteration_time_ms`
- `estimate.estimator_source`
- `estimate.forward_ms`
- `estimate.backward_compute_ms`
- `estimate.backward_comm_ms`
- `estimate.microbatch_slot_ms`
- `estimate.optimizer_step_ms`
- `phase_estimates`
- `train_infer_static_estimate`
- `profile_estimate`
- `comparison.train_iteration_relative_error_pct`
- `rank_measurements.train_iteration`

## 7. 默认验证入口

单卡 baseline：

```bash
bash scripts/run_cambricon_train_tp_single.sh
```

两卡 TP：

```bash
bash scripts/run_cambricon_train_tp_multi.sh
```

完整 smoke：

```bash
bash scripts/run_cambricon_train_smoke.sh
```

根目录入口：

```bash
bash scripts/run_training_single_cambricon.sh
bash scripts/run_training_multi_cambricon.sh
bash scripts/run_training_cambricon_smoke.sh
```
