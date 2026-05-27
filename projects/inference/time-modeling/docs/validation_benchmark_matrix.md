# 验证基准矩阵

源码对应关系见 `docs/source_traceability.md`。

## 1. 目标

验证基准矩阵用于回答两个问题：

1. 预测器在哪些模型/场景上已经可信；
2. 误差主要来自哪些维度：shape、dtype、并行方式、通信、运行时开销，还是 decode loop 语义。

矩阵应服务于 v1 开发节奏，而不是一开始就覆盖所有模型家族。


## 2. 验证指标

每个 benchmark case 至少报告：

- `measured_end_to_end_time_ms`
- `predicted_end_to_end_time_ms`
- `abs_error_ms`
- `relative_error`
- `mape`
- `component_errors`
- `topk_bottleneck_match`

训练侧额外报告：

- `forward_error`
- `backward_error`
- `optimizer_error`
- `gradient_sync_error`

推理侧额外报告：

- `prefill_error`
- `decode_step_error`
- `request_e2e_error`


## 3. 分阶段矩阵策略

### 3.1 P0: 功能打通

目标：证明 schema、profile、预测、验证链路可运行。

- 模型规模：1B 级或更小；
- 设备：单 GPU；
- 模式：推理 + 训练；
- 并行：无分布式；
- dtype：bf16 或 fp16。

### 3.2 P1: 核心精度

目标：覆盖主流 dense decoder-only LLM。

- 模型规模：1B、7B/8B；
- 设备：单 GPU + 2/4 GPU；
- 模式：推理 prefill/decode、训练 forward/backward/update；
- 并行：DP、TP 基础场景；
- 序列长度：短、中、长。

### 3.3 P2: 扩展场景

目标：覆盖更复杂 family 和更长上下文。

- MoE；
- 多节点通信；
- 更长 context；
- 复杂 optimizer/ZeRO；
- 替代运行时模型。


## 4. 模型集合

### 4.1 v1 必测模型

建议最小集合：

- `TinyLlama-1.1B` 或同级小模型
  - 低成本打通推理/训练链路；
- `Llama-3-8B-Instruct` 或同级 7B/8B dense decoder-only 模型
  - 作为主验证对象；
- `Qwen2.5-7B` 或同级模型
  - 检验不同实现风格下的泛化。

### 4.2 v1 可选模型

- `Mistral-7B`
- `Llama-2-7B`

### 4.3 v2 候选模型

- `Mixtral` / MoE 模型
- 更长上下文模型
- 非 Transformer 家族模型


## 5. 硬件矩阵

### 5.1 v1 必测硬件

- `A100 80G`
  - 作为稳定基线；
- 当前实际开发 GPU
  - 作为本地迭代目标。

### 5.2 v1 可选硬件

- `H100 80G`
- `4090` / `L40S` 一类单机 GPU

### 5.3 多卡拓扑

v1 建议至少覆盖：

- 单机 2 GPU
- 单机 4 GPU

若有条件，再补：

- 双机 8 GPU


## 6. 推理基准矩阵

### 6.1 单 GPU 推理

| Case ID | Model | dtype | bs | isl | osl | 目标 |
|---|---|---:|---:|---:|---:|---|
| INF-SG-01 | TinyLlama-1.1B | bf16 | 1 | 128 | 64 | 打通 prefill/decode |
| INF-SG-02 | TinyLlama-1.1B | bf16 | 1 | 2048 | 128 | 长 prompt |
| INF-SG-03 | Llama-3-8B | bf16 | 1 | 128 | 64 | 主干模型短序列 |
| INF-SG-04 | Llama-3-8B | bf16 | 1 | 2048 | 128 | 主干模型中长序列 |
| INF-SG-05 | Llama-3-8B | bf16 | 4 | 2048 | 128 | batch 效应 |
| INF-SG-06 | Qwen2.5-7B | bf16 | 1 | 4096 | 128 | 家族泛化 |

### 6.2 多 GPU 推理

| Case ID | Model | dtype | Parallel | bs | isl | osl | 目标 |
|---|---|---:|---|---:|---:|---:|---|
| INF-MG-01 | Llama-3-8B | bf16 | TP=2 | 1 | 2048 | 128 | 基础张量并行 |
| INF-MG-02 | Llama-3-8B | bf16 | TP=4 | 1 | 2048 | 128 | 通信主导检查 |
| INF-MG-03 | Qwen2.5-7B | bf16 | TP=2 | 4 | 2048 | 128 | batch + comm |

推理侧必须对每个 case 同时验证：

- `prefill_end_to_end_time`
- `decode_step_end_to_end_time`
- `request_end_to_end_time`


## 7. 训练基准矩阵

### 7.1 单 GPU 训练

| Case ID | Model | dtype | mbs | seq | GA | Optimizer | 目标 |
|---|---|---:|---:|---:|---:|---|---|
| TR-SG-01 | TinyLlama-1.1B | bf16 | 1 | 512 | 1 | AdamW | 打通 forward/backward/update |
| TR-SG-02 | TinyLlama-1.1B | bf16 | 2 | 2048 | 1 | AdamW | 长序列训练 |
| TR-SG-03 | Llama-3-8B | bf16 | 1 | 1024 | 1 | AdamW | 主干模型单卡 |
| TR-SG-04 | Llama-3-8B | bf16 | 1 | 2048 | 4 | AdamW | 梯度累加 |

### 7.2 多 GPU 训练

| Case ID | Model | dtype | Parallel | mbs | seq | GA | 目标 |
|---|---|---:|---|---:|---:|---:|---|
| TR-MG-01 | Llama-3-8B | bf16 | DDP=2 | 1 | 1024 | 1 | 基础梯度同步 |
| TR-MG-02 | Llama-3-8B | bf16 | DDP=4 | 1 | 2048 | 1 | gradient sync 精度 |
| TR-MG-03 | Llama-3-8B | bf16 | TP=2 | 1 | 2048 | 1 | activation comm |
| TR-MG-04 | Llama-3-8B | bf16 | TP=2, DP=2 | 1 | 2048 | 4 | 复合并行 |

训练侧必须对每个 case 同时验证：

- `forward_compute_time`
- `backward_compute_time`
- `gradient_sync_time`
- `optimizer_time`
- `train_iteration_time`


## 8. 维度扫描矩阵

除固定 case 外，建议对主干模型做维度扫描：

### 8.1 推理扫描

- `batch_size`: 1 / 2 / 4 / 8
- `isl`: 128 / 512 / 2048 / 4096 / 8192
- `osl`: 32 / 128 / 512
- `dtype`: bf16 / fp16

### 8.2 训练扫描

- `micro_batch_size`: 1 / 2 / 4
- `seq_len`: 512 / 1024 / 2048 / 4096
- `GA`: 1 / 4 / 8
- `checkpointing`: off / on


## 9. 接受阈值

### 9.1 v1 建议阈值

单 GPU：

- 端到端 `MAPE <= 15%`
- 主要组件项 `MAPE <= 20%`

多 GPU：

- 端到端 `MAPE <= 20%`
- 通信相关组件 `MAPE <= 25%`

### 9.2 瓶颈一致性阈值

- top-5 瓶颈算子中至少 `3/5` 排名一致；
- stage-level 主瓶颈判断不能反转。


## 10. 验证执行协议

每个 benchmark case 建议执行顺序：

1. 生成或加载数据库；
2. 生成预测结果；
3. 真实运行 Torch eager；
4. 采集阶段级和端到端实测；
5. 写入 `ValidationRecord`；
6. 汇总生成 case table、误差图和 bottleneck 对比。


## 11. 最小输出报表

每次验证批次至少输出：

- `validation_summary.md`
- `validation_results.csv`
- `component_error_breakdown.csv`
- `bottleneck_match.csv`


## 12. 与参考仓库的关系

- 借鉴 nnScaler / nnScaler-M 对训练侧 compute / comm / weight-update 分项的关注点；
- 借鉴 AIConfigurator 对表格化 summary 和可视化友好输出的组织方式；
- 不把 serving SLA、worker autoscaling 之类指标纳入 v1 核心 benchmark matrix。
