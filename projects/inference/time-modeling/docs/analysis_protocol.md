# 分析协议文档

源码对应关系见 `docs/source_traceability.md`。

## 1. 目标

本协议定义如何为 Torch-first LLM 时间预测系统生成可信的 profile 数据。

协议的目标不是追求一次性覆盖所有算子，而是保证：

- profile 数据可复现；
- 采集边界与时间语义一致；
- 可以从单算子、模块、通信三种粒度逐步扩展；
- 采集结果可直接进入代价数据库。


## 2. 分析对象

分析对象分三类：

1. `op profiling`
   - 单 ATen / Torch op；
2. `module profiling`
   - attention block、MLP block、Transformer block、optimizer substep；
3. `communication profiling`
   - all-reduce、all-gather、reduce-scatter、broadcast、p2p、all-to-all。


## 3. 公共环境要求

每次 profile 必须记录环境快照：

- GPU 型号与显存；
- CUDA 版本；
- PyTorch 版本；
- NCCL 版本；
- 驱动版本；
- 运行时模型版本；
- 是否开启 `torch.compile`；
- 是否开启 activation checkpoint；
- 分布式 world size / rank 布局；
- 固定随机种子；
- 电源/功耗模式（如可获取）。

默认要求：

- 独占 GPU；
- 无其他显著负载；
- 采样前完成 CUDA 上下文初始化；
- 明确使用 `torch.cuda.synchronize()` 保证测量边界。


## 4. 图到 profile key 的生成流程

### 4.1 图提取

从 `torch.fx`、`torch.export` 或 eager trace 中提取原始节点。

### 4.2 归一化

对每个节点生成稳定 `query_key`，至少包括：

- `op_type`
- `phase`
- `shape_signature`
- `dtype_signature`
- `requires_grad`
- `layout_signature`
- `distributed_signature`

### 4.3 去重

同一 graph 内对于相同 `query_key` 的节点只需 profile 一次。

这一步直接借鉴 nnScaler 的做法，但 key 生成必须切换到新的 IR schema。


## 5. 单算子分析协议

### 5.1 输入构造

对于每个 `query_key`：

- 构造与 shape/dtype 相匹配的 dummy tensors；
- 保持 `requires_grad` 状态与原图一致；
- 参数、buffer、activation 的角色必须保留；
- 非张量参数必须按原始属性传入；
- 若某 op 对值域敏感，允许使用 op-specific input generator。

### 5.2 预热

默认：

- 预热时间不少于 `2s`，或不少于 `20` 次执行；
- 直到单次执行时间稳定进入可接受波动范围；
- 预热过程不写入数据库。

### 5.3 正式测量

建议至少采集：

- `forward only`
- `forward + backward`
- 可选 `recompute replay`

默认采集统计：

- `mean`
- `median`
- `std`
- `p95`
- `measured_runs`

### 5.4 内存相关采集

建议至少采集：

- `input_bytes`
- `param_bytes`
- `buffer_bytes`
- `saved_activation_bytes`
- `peak_infer_bytes`
- `peak_train_bytes`

保存 activation 的识别方式可借鉴 nnScaler 中 `saved_tensors_hooks` 的思路。

### 5.5 失败处理

若 profile 失败：

- 记录失败原因；
- 不直接写入伪测量值作为 measured record；
- 允许写入 `fallback_required=true` 的占位记录；
- 由后续 module-level 或 analytic fallback 兜底。


## 6. 模块级分析协议

### 6.1 启用条件

当满足以下情况之一时，允许或建议使用模块级 profile：

- 单算子难以代表真实执行；
- Torch eager 中存在明显融合/交互；
- decode step 中 KV cache 读写与注意力逻辑强耦合；
- 优化器子步骤需要整体建模。

### 6.2 最小模块集

v1 推荐优先支持：

- `attention_block_prefill`
- `attention_block_decode_step`
- `mlp_block`
- `transformer_block_forward`
- `transformer_block_backward`
- `optimizer_substep`

### 6.3 模块 profile 记录

模块 profile 必须额外记录：

- `covered_node_ids`
- `covered_op_families`
- `substitution_policy`

即：它在预测时替代哪些原子节点。


## 7. 通信分析协议

### 7.1 原语范围

v1 至少覆盖：

- all-reduce
- all-gather
- reduce-scatter
- broadcast
- p2p send/recv

若模型或并行策略需要，再扩展 all-to-all。

### 7.2 采样维度

每个通信原语应按以下维度采样：

- `world_size`
- `mesh_shape`
- `message_size`
- `dtype`
- `topology`
- `intra_node` / `inter_node`

### 7.3 采样点

消息大小建议按对数尺度采样，覆盖：

- 小消息；
- 中等消息；
- 大消息；
- 接近显存/带宽上限的长消息。

### 7.4 记录内容

每条通信 profile 记录至少包含：

- `latency_us`
- `bandwidth_gbps`
- `fit_kind`
- `message_sizes_bytes`
- `repetitions`

### 7.5 插值与外推

- 小范围内允许 piecewise linear 插值；
- 超出采样范围时必须标记 `evidence_level=extrapolated`；
- 大消息外推可采用带宽上限线性外推，但必须记录风险。


## 8. 训练专项协议

### 8.1 microbatch 与 optimizer step 必须分开采集

训练 profile 至少需要两层：

- microbatch forward/backward；
- optimizer update。

### 8.2 梯度累加

采样时必须记录：

- `gradient_accumulation_steps`
- 梯度同步是在每个 microbatch 还是在 update 边界发生。

### 8.3 重计算

若启用 checkpoint/recompute：

- 必须单独记录 `recompute_time`；
- 不允许把重计算时间直接塞回原始 forward profile 中而不做区分。


## 9. 推理专项协议

### 9.1 必须分离 prefill 与 decode

即使最终对外只给一个请求总时间，原始采集也必须分离：

- `prefill`
- `decode_step`

### 9.2 KV cache

推理 profile 必须显式记录：

- 是否使用 KV cache；
- `kv_cache_dtype`；
- `num_heads` / `num_kv_heads`；
- `head_dim`；
- decode step 中累计上下文长度。

### 9.3 序列长度采样

prefill 与 decode 的采样策略应不同：

- prefill 重点覆盖 `isl`；
- decode 重点覆盖 `current_context_len` 和 batch/token 数。


## 10. 质量门槛

profile 结果进入数据库前，建议至少满足：

- `measured_runs >= 10`；
- `cov <= 0.1`，否则标记为高噪声；
- 明确成功/失败状态；
- 具备完整环境元数据；
- 时间单位统一。


## 11. 产物清单

每次分析批次建议输出：

- `manifest.json`
- `compute_records.jsonl`
- `comm_records.jsonl`
- `failures.jsonl`
- `summary.md`
- 可选 `raw_traces/`


## 12. 与验证流程的衔接

分析协议的目标不是单独完成验证，而是为验证提供高质量基础数据。

后续验证阶段应：

1. 使用本协议生成数据库；
2. 对 benchmark matrix 中的案例运行预测；
3. 再运行真实 Torch eager 执行；
4. 产出 `ValidationRecord`。


## 13. 与参考仓库的关系

- 继承 nnScaler / nnScaler-M 在 CUDA 实测、forward/backward 拆分、通信离线 profile 上的经验；
- 继承 AIConfigurator 在按 op family 组织数据库和 summary 的工程化表达方式；
- 不沿用面向特定 serving backend 的 profile 维度作为核心协议。
