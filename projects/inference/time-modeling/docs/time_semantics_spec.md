# 时间语义规范

源码对应关系见 `docs/source_traceability.md`。

## 1. 目标

本规范定义 Torch-first LLM 时间预测系统中所有时间相关名词、单位、边界和组合规则。
目标是避免将不同来源的时间值混用，例如：

- 把单算子 profile 时间与端到端请求时间直接比较；
- 把 microbatch 时间与 optimizer step 时间直接相加；
- 把服务系统的排队时间、调度时间和纯 Torch eager 执行时间混为一谈。

本规范优先面向 `torch_eager` 运行时模型 v1，但命名应尽量可复用于后续 `torch_compile`、CUDA Graph 或其他运行时模型。


## 2. 总体原则

### 2.1 单位

- 规范单位：时间统一使用 `ms` 作为对外展示单位；
- 数据库存储允许使用 `us`、`ms` 或 `s`，但必须带显式 `unit` 字段；
- 内部组合时允许统一换算为 `us` 以减少浮点误差；
- 最终 API/报告默认输出 `ms`。

### 2.2 时间类型

系统中的时间值必须标注其语义类型之一：

- `measured_time`: 真实测量值；
- `profiled_time`: 离线 profile 获得的局部测量值；
- `estimated_time`: 由数据库查询和组合规则得到的预测值；
- `calibrated_time`: 在估计值基础上经过修正层得到的预测值。

### 2.3 时间边界

所有时间值必须说明是否包含：

- 计算；
- 通信；
- 框架开销；
- 内存相关 stall 修正；
- 优化器/参数更新；
- 重计算；
- 排队或服务层调度。

默认情况下，本项目 v1 不包含服务系统排队时间，不建模多请求队列调度。


## 3. 规范命名

### 3.1 原子级指标

- `op_forward_compute_time`
  - 单算子前向计算时间；
  - 不含跨设备通信；
  - 可含该算子内部不可分离的 kernel launch 开销。

- `op_backward_compute_time`
  - 单算子反向计算时间；
  - 仅训练图有意义。

- `op_recompute_time`
  - 由于 activation checkpoint/recompute 引入的额外前向重放时间；
  - 不与 `op_forward_compute_time` 重复记账。

- `op_runtime_overhead_time`
  - 归属于算子的 Torch eager 框架开销修正；
  - 如 Python 调度、张量包装、同步伪影等；
  - 只能由修正层注入，不能冒充 profile 原值。

- `edge_comm_time`
  - 图边上的通信时间；
  - 对应 activation、parameter、gradient 或 control token 的跨设备传输。

- `op_memory_stall_correction_time`
  - 因单算子 profile 难以表达的内存系统饱和、cache miss、allocator 抖动等所加的修正项；
  - v1 中允许为 0。

### 3.2 区域级指标

- `subgraph_compute_time`
  - 一个子图或 region 内所有计算项之和；
- `subgraph_comm_time`
  - 一个子图或 region 内所有通信项之和；
- `subgraph_runtime_overhead_time`
  - 一个子图或 region 的运行时修正项之和；
- `critical_path_time`
  - 在依赖与重叠规则下的关键路径时间；
- `sum_time`
  - 单纯加和时间，不考虑重叠，仅用于调试，不作为端到端时间。

### 3.3 训练级指标

- `forward_compute_time`
  - 一个训练 microbatch 的前向计算时间总和；
- `backward_compute_time`
  - 一个训练 microbatch 的反向计算时间总和；
- `activation_comm_time`
  - 前向和反向因 activation 分片/聚合/传递导致的通信时间；
- `gradient_sync_time`
  - 梯度同步时间；
- `parameter_sync_time`
  - 参数 gather、broadcast、ZeRO/FSDP all-gather 等参数相关通信时间；
- `optimizer_time`
  - 优化器数学更新和状态读写时间；
- `runtime_overhead_time`
  - 不属于前述类别的训练期运行时开销；
- `microbatch_step_time`
  - 单个 microbatch 的端到端时间；
- `optimizer_step_time`
  - 一个 optimizer 更新周期的时间；
- `train_iteration_time`
  - 对外报告的训练步时间，默认等于一次 optimizer 更新周期时间。

### 3.4 推理级指标

- `prefill_compute_time`
  - prompt/prefill 图的计算时间；
- `prefill_comm_time`
  - prefill 阶段的通信时间；
- `prefill_runtime_overhead_time`
  - prefill 阶段的框架修正项；
- `prefill_end_to_end_time`
  - prefill 阶段端到端时间；
- `decode_step_compute_time`
  - 单个 decode step 的计算时间；
- `decode_step_comm_time`
  - 单个 decode step 的通信时间；
- `decode_step_runtime_overhead_time`
  - 单个 decode step 的框架修正项；
- `decode_step_end_to_end_time`
  - 单个 decode step 的端到端时间；
- `request_end_to_end_time`
  - 单个请求从开始执行到完成生成的时间；
- `graph_compute_time`
  - 对于不显式拆成 prefill/decode 的图，使用统一命名表示图内总计算时间；
- `graph_comm_time`
  - 同上，图内总通信时间；
- `end_to_end_time`
  - 统一的顶层端到端时间别名。


## 4. 推理时间语义

### 4.1 基本执行对象

在 `torch_eager` v1 中，推理默认拆成两个执行对象：

- `prefill graph`
  - 处理输入 prompt；
- `decode step graph`
  - 处理每个增量 token 的一步前向。

如果模型或执行方式不适合拆分，也允许退化为单一 `inference graph`。

### 4.2 request 级时间定义

对于输入长度 `isl`、输出长度 `osl` 的单请求：

- 若 `osl == 0`，则只报告 prefill；
- 若 `osl >= 1`，默认：

```text
request_end_to_end_time
  = prefill_end_to_end_time
  + sum(decode_step_end_to_end_time[i] for i in generated_steps)
```

若 v1 只建模平均 decode step，则使用：

```text
request_end_to_end_time
  = prefill_end_to_end_time
  + generated_steps * mean_decode_step_end_to_end_time
```

其中：

- `generated_steps` 通常等于 `max(osl - 1, 0)`；
- 是否把首 token 计入 prefill 还是计入 decode，必须由运行时模型统一定义；
- v1 规定：首 token 归入 prefill 完成时刻，因此 `generated_steps = max(osl - 1, 0)`。

### 4.3 TTFT / TPOT 的位置

- `ttft` 和 `tpot` 不是核心内部规范名词；
- 它们仅作为派生展示指标：

```text
ttft = prefill_end_to_end_time
tpot = mean_decode_step_end_to_end_time
```

服务系统相关的 queueing、batching、continuous batching 调度不属于 v1 核心定义。


## 5. 训练时间语义

### 5.1 基本执行对象

训练侧必须区分三个层级：

- `microbatch`
  - 一次前向 + 反向；
- `optimizer update`
  - 完成一次参数更新；
- `train iteration`
  - 对外报告的训练步时间。

### 5.2 默认定义

若存在梯度累加，设：

- `gradient_accumulation_steps = G`
- 每个 optimizer step 含 `G` 个 microbatch

则默认：

```text
optimizer_step_time
  = sum(microbatch_step_time[g] for g in 1..G)
  + optimizer_time_extra
```

其中 `microbatch_step_time` 可定义为：

```text
microbatch_step_time
  = forward_compute_time
  + backward_compute_time
  + activation_comm_time
  + gradient_sync_time_if_not_deferred
  + parameter_sync_time_if_inline
  + runtime_overhead_time
  + op_recompute_time
```

若梯度同步被延迟到 optimizer step 结束时统一发生，则：

- microbatch 内不计 `gradient_sync_time`；
- optimizer step 级单独记 `gradient_sync_time`。

默认对外主指标：

```text
train_iteration_time = optimizer_step_time
```

### 5.3 优化器的边界

`optimizer_time` 默认包含：

- 参数读取；
- 梯度读取；
- 优化器状态更新；
- 参数写回；
- 与优化器强绑定的 cast / scale / unscale / clip。

`optimizer_time` 默认不包含：

- dataloader；
- checkpoint 保存；
- 日志打印；
- 外部控制逻辑。


## 6. 组合规则

### 6.1 sum 与 critical path 必须并存

所有预测结果至少应同时保留：

- `sum_time`: 所有组成项简单求和；
- `critical_path_time`: 按依赖和重叠规则求得的关键路径时间。

最终端到端时间默认取 `critical_path_time`，不是 `sum_time`。

### 6.2 重叠规则

如果某一运行时模型声明通信与计算可重叠，则需要显式记录：

- `overlap_eligible = true/false`
- `overlap_group_id`
- `overlap_credit_time`

v1 可先采用保守策略：

- 默认不假设跨算子重叠；
- 仅在运行时模型明确声明时启用有限重叠。


## 7. 测量和预测的一致性要求

### 7.1 测量必须说明样本边界

任意测量值都必须记录：

- execution unit: `op`, `subgraph`, `prefill`, `decode_step`, `microbatch`, `optimizer_step`, `train_iteration`；
- repetition count；
- warmup 策略；
- aggregation method：`mean`, `median`, `p50`, `p95` 等；
- whether synchronized。

### 7.2 预测必须说明证据等级

预测输出建议记录：

- `evidence_level = measured | interpolated | extrapolated | calibrated | fallback`；
- `source_record_ids`；
- `confidence_band` 或 `uncertainty_ratio`。


## 8. 命名兼容策略

参考仓库中存在如下旧命名：

- `fw_span` / `bw_span`；
- `comp_time`；
- `weight_update_time`；
- `all_time`；
- `context_latency` / `generation_latency`；
- `ttft` / `tpot`。

在新系统中：

- 可在导入层保留兼容映射；
- 对外文档统一使用本规范中的标准名；
- 不再将 `all_time` 作为无语义解释的总时间主名词。


## 9. v1 强制约束

v1 必须满足：

- 所有时间字段可追溯到规范名词；
- 推理与训练必须使用不同的顶层时间定义；
- 微批、优化器步、请求级时间不得混用；
- 输出必须显式区分 `measured` 与 `estimated`；
- 端到端时间必须可分解为组件项和关键路径解释。


## 10. 建议的最小输出字段

### 10.1 推理

```json
{
  "runtime_model": "torch_eager_v1",
  "mode": "inference",
  "prefill_end_to_end_time_ms": 0.0,
  "decode_step_end_to_end_time_ms": 0.0,
  "request_end_to_end_time_ms": 0.0,
  "graph_compute_time_ms": 0.0,
  "graph_comm_time_ms": 0.0,
  "runtime_overhead_time_ms": 0.0,
  "critical_path_time_ms": 0.0
}
```

### 10.2 训练

```json
{
  "runtime_model": "torch_eager_v1",
  "mode": "training",
  "forward_compute_time_ms": 0.0,
  "backward_compute_time_ms": 0.0,
  "activation_comm_time_ms": 0.0,
  "gradient_sync_time_ms": 0.0,
  "parameter_sync_time_ms": 0.0,
  "optimizer_time_ms": 0.0,
  "runtime_overhead_time_ms": 0.0,
  "train_iteration_time_ms": 0.0,
  "critical_path_time_ms": 0.0
}
```


## 11. 与参考仓库的关系

- 吸收 `nnscaler` / `nnscaler-M` 中对 `fw_span`、`bw_span`、`comp_time`、`weight_update_time` 的拆分思路；
- 吸收 AIConfigurator 中 phase-level 聚合与 summary schema 的表达方式；
- 不直接继承服务系统语义作为本系统核心时间语义。
