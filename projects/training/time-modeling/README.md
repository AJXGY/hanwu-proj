# Training Time Modeling

寒武纪训练时间维度验证工作区。当前训练主线已切到 Tensor Parallel，参考
`/home/o_mabin/train-infer-estimation` 的训练估算思路：用静态 FLOPs/bytes + 校准得到估算值，
输出 `T_real`、`T_sim` 和误差，并把训练 iteration 拆成 `forward / backward_compute / backward_comm / optimizer`
四个可解释阶段。

## 当前目标

1. 训练并行主线使用 `single / tp`，不再以 `pp_size` 作为默认训练路线。
2. 支持 `tp_size=1/2`，其中 `tp_size=1` 作为单卡 baseline，`tp_size=2` 作为两卡 TP。
3. 支持真实 `Llama-3.1-8B` 在寒武纪 MLU 上跑训练 smoke。
4. 误差目标：`train_iteration_relative_error_pct < 20%`。
5. 默认 `T_sim` 来自 `train_infer_static`；训练 profile 写入 JSONL 只作为对照和调参材料。

## 核心入口

- `torch_train_tp_mvp.py`: TP 训练主入口。
- `tp_backend.py`: CUDA/MLU 设备、分布式后端、同步和拓扑辅助。
- `scripts/run_cambricon_train_tp_single.sh`: 单卡训练 baseline。
- `scripts/run_cambricon_train_tp_multi.sh`: 两卡 TP 训练 smoke。
- `scripts/run_cambricon_train_smoke.sh`: 先跑 single baseline，再跑 TP2 smoke。
- `train_dashboard.py`: 训练 Dashboard，默认调用 TP 主入口。

## 时间语义

训练主指标：

```text
T_real = measured train iteration wall time
T_sim  = microbatch_count * (forward + backward_compute + backward_comm) + optimizer_step
error  = abs(T_real - T_sim) / T_real * 100%
```

当前 report 中保留 `train-infer-estimation` 风格的结构：

- `execution.parallel_mode`
- `execution.tp_size`
- `execution.estimator_source`
- `measured.train_iteration_time_ms`
- `estimate.train_iteration_time_ms`
- `estimate.forward_ms`
- `estimate.backward_compute_ms`
- `estimate.backward_comm_ms`
- `estimate.microbatch_slot_ms`
- `estimate.optimizer_step_ms`
- `phase_estimates`
- `comparison.train_iteration_relative_error_pct`
- `train_infer_static_estimate`
- `profile_estimate`
- `rank_measurements.train_iteration`

## TP 实现策略

`torch_train_tp_mvp.py` 使用 `torch.distributed.tensor.parallel`：

- `ColwiseParallel`: `q/k/v/gate/up` projection。
- `RowwiseParallel`: `o/down` projection。
- `torch.distributed.run`: 启动 `tp_size=2` 的多进程训练。
- MLU 后端使用 `cncl`，CUDA 后端使用 `nccl`。

训练负载已对齐 `train-infer-estimation` 的 `lora_vocab_adapter`：

- Llama backbone 不冻结，forward 不包 `no_grad`，hidden state 不 `detach`。
- autograd 穿过完整 backbone，backward workload 与参考项目对齐。
- adapter 是 `hidden_size -> rank -> vocab_size` 的 token-level vocab head。
- loss 是 token-level vocab cross entropy，labels 使用 synthetic `input_ids`。
- optimizer 只更新 adapter 参数。
- 默认优化器 `sgd`。
- 默认开启 `gradient checkpointing`。

这不再是 2 类 classifier 任务；`Llama-3.1-8B`、`lora_rank=8` 时 adapter 参数量为
`8 * (4096 + 128256) = 1,058,816`。

## 快速运行

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

根目录统一入口：

```bash
bash scripts/run_training_single_cambricon.sh
bash scripts/run_training_multi_cambricon.sh
bash scripts/run_training_cambricon_smoke.sh
```

Dashboard：

```bash
bash scripts/run_cambricon_train_dashboard.sh
```

默认地址：

```text
http://127.0.0.1:8234
```

## 默认模型和校准

默认真实模型目录：

```text
/home/o_mabin/LLM/models/Llama-3.1-8B
```

默认静态 estimator 配置：

```text
configs/train_config.yaml
```

默认 profile 表仍保留为对照：

```text
database/train_component_profile_cambricon_mlu580_tp_single.jsonl
database/train_component_profile_cambricon_mlu580_tp2.jsonl
```

可通过环境变量覆盖：

```bash
HOST_MODEL_DIR=/path/to/model bash scripts/run_cambricon_train_tp_multi.sh
SEQUENCE_LENGTH=64 MICROBATCH_COUNT=1 bash scripts/run_cambricon_train_tp_multi.sh
OPTIMIZER_TYPE=adamw bash scripts/run_cambricon_train_tp_multi.sh
```

## 已验证结果

在当前 `hanwu-proj` 目录中运行：

```bash
SEQUENCE_LENGTH=8 MICROBATCH_COUNT=1 bash scripts/run_training_cambricon_smoke.sh
```

得到：

- single baseline: `T_real=111.367228 ms`，`T_sim=105.804344 ms`，`error=4.995081%`
- TP2: `T_real=319.946518 ms`，`T_sim=314.767113 ms`，`error=1.618835%`

两组都低于 `20%` 误差目标，且本次 `T_sim` 的 implementation 为 `mvp_train_unified_estimator`。
