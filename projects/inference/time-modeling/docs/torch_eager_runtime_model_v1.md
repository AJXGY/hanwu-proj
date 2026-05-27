# torch_eager 运行时模型 v1

源码对应关系见 `docs/source_traceability.md`。

## 1. 定位

`torch_eager_v1` 是本项目的第一个正式运行时模型。

它回答的问题不是：

- 一个生产级 serving 系统在多请求队列下的吞吐/延迟；
- 一个编译后端的理想 kernel 调度时间；
- 一个周期级模拟器的精确执行轨迹。

它回答的问题是：

- 在普通 Torch eager 风格执行下，这个 LLM 图、子图或训练步骤大概要花多久；
- 时间主要分布在哪些算子、region、阶段和通信项上。


## 2. v1 假设

### 2.1 共同假设

- 执行模式为 PyTorch eager；
- 默认单进程单设备，或一进程一设备的基础分布式执行；
- 不假设 CUDA Graph；
- 不假设 `torch.compile`；
- 默认使用标准 CUDA stream 语义；
- 不建模服务系统队列、请求调度、异步 worker 池；
- 不建模 dataloader 和 checkpoint I/O。

### 2.2 允许建模的内容

- kernel 计算时间；
- collectives 与 p2p 通信；
- Torch eager 运行时开销修正；
- activation checkpoint / recompute；
- optimizer step；
- 基础计算/通信重叠规则。


## 3. 输入契约

`torch_eager_v1` 运行时模型的最小输入包括：

- 归一化 `GraphSpec`；
- `execution_config`；
- `device_mesh`；
- 代价数据库句柄；
- 可选修正层。

### 3.1 推理最小输入

- `batch_size`
- `seq_len`
- `output_len`
- `dtype`
- `kv_cache_dtype`
- `device_mesh`

### 3.2 训练最小输入

- `micro_batch_size`
- `seq_len`
- `dtype`
- `gradient_accumulation_steps`
- `optimizer_type`
- `checkpointing`
- `parallel_config`


## 4. 推理模型

### 4.1 结构

推理默认拆成：

- `prefill region`
- `decode_step region`

其中 `decode_step region` 通过 `loop_region` 表示重复执行。

### 4.2 预测流程

1. 对 prefill region 中每个节点查询 `ComputeProfileRecord`；
2. 对显式通信边查询 `CommProfileRecord`；
3. 按 region 依赖关系组合 `compute + comm + runtime_overhead`；
4. 计算 `prefill_end_to_end_time`；
5. 对 decode body 在给定 `current_context_len` 下查询节点 profile；
6. 组合出 `decode_step_end_to_end_time`；
7. 根据 `output_len` 聚合为请求级总时间。

### 4.3 KV cache 语义

v1 要求 decode step 图显式表达：

- 读取已有 KV cache；
- 写回新增 KV cache；
- 当前上下文长度对 attention kernel 的影响。

因此 decode step 的 query key 不能只依赖 `batch_size`，还必须依赖：

- `current_context_len`
- `num_heads`
- `num_kv_heads`
- `head_dim`
- `kv_cache_dtype`

### 4.4 端到端公式

```text
request_end_to_end_time
  = prefill_end_to_end_time
  + sum(decode_step_end_to_end_time[i])
```

v1 若没有 step-wise 曲线，可退化为：

```text
request_end_to_end_time
  = prefill_end_to_end_time
  + generated_steps * mean_decode_step_end_to_end_time
```


## 5. 训练模型

### 5.1 结构

训练默认拆成：

- `forward region`
- `backward region`
- `optimizer region`
- 可选 `comm regions`

### 5.2 预测流程

1. 对 forward 节点查询前向 profile；
2. 对 backward 节点查询反向 profile；
3. 若开启 recompute，则附加 `recompute_time`；
4. 对 activation / gradient / parameter 通信边查询通信 profile；
5. 对 optimizer region 查询模块级或算子级 profile；
6. 聚合为 `microbatch_step_time`；
7. 按 `gradient_accumulation_steps` 聚合为 `optimizer_step_time`；
8. 输出 `train_iteration_time`。

### 5.3 通信项分类

吸收 nnScaler-M 的经验，训练侧通信不合并成单一 `comm_time`，而至少分成：

- `activation_comm_time`
- `gradient_sync_time`
- `parameter_sync_time`

如果未来要兼容 ZeRO / FSDP，可继续细分。


## 6. 组合引擎

### 6.1 节点级组合

每个节点的预测值由以下项组成：

```text
node_time
  = op_forward_compute_time
  + op_backward_compute_time
  + op_recompute_time
  + op_runtime_overhead_time
  + op_memory_stall_correction_time
```

具体启用哪些项取决于 phase。

### 6.2 边级组合

每条通信边的预测值：

```text
edge_time = edge_comm_time
```

### 6.3 关键路径

默认端到端时间采用关键路径，而非纯加和：

```text
end_to_end_time = critical_path(node_times, edge_times, overlap_rules)
```

### 6.4 v1 重叠策略

v1 采用保守策略：

- 默认不假设跨节点重叠；
- 对显式声明的通信-计算配对，允许有限重叠；
- 修正项必须透明输出，不允许在总时间中“隐式吃掉”。


## 7. 修正层

`torch_eager_v1` 允许一个轻量修正层，但必须满足：

- 修正与原始 profile 分离存储；
- 修正值可追溯；
- 修正后的结果要保留 `raw_estimate` 与 `calibrated_estimate`。

建议优先支持：

- `global_runtime_overhead_bias`
- `op_family_scale`
- `decode_seq_len_bucket_scale`
- `training_optimizer_bias`


## 8. 输出契约

### 8.1 推理输出

至少应输出：

- `prefill_end_to_end_time`
- `decode_step_end_to_end_time`
- `request_end_to_end_time`
- `graph_compute_time`
- `graph_comm_time`
- `runtime_overhead_time`
- `critical_path_time`
- `top_ops`
- `top_regions`

### 8.2 训练输出

至少应输出：

- `forward_compute_time`
- `backward_compute_time`
- `activation_comm_time`
- `gradient_sync_time`
- `parameter_sync_time`
- `optimizer_time`
- `runtime_overhead_time`
- `train_iteration_time`
- `critical_path_time`
- `top_ops`
- `top_regions`


## 9. v1 不支持或仅部分支持的内容

- continuous batching 服务调度；
- 多请求 queueing；
- 高级 overlap 精确建模；
- 自动分区搜索本身；
- 非 Torch eager 主运行时；
- 任意动态控制流。


## 10. v1 与参考仓库的关系

- 从 nnScaler / nnScaler-M 复用训练侧的分项思路：compute、activation comm、weight/gradient update communication；
- 从 AIConfigurator 复用 phase-level 分解、summary object、可视化友好输出思路；
- 不继承 AIConfigurator 的 serving worker / autoscaling / SLA 搜索语义；
- 不要求用户先将模型转换成 nnScaler 专有 IR 才能使用预测器。


## 11. 向 v2 的自然扩展

`torch_eager_v1` 之后最自然的扩展方向：

1. `torch_compile_inductor_v1`
2. 更强的 decode step-by-step 曲线模型
3. 更明确的通信-计算重叠规则
4. 更丰富的分布式推理模型
5. 自动校准和可视化报告
