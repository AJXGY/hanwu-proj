# 内部图 IR 模式

源码对应关系见 `docs/source_traceability.md`。

## 1. 设计目标

内部 IR 负责把 Torch 前端图、nnScaler 风格图和后续运行时模型之间的耦合拆开。

该 IR 必须：

- 同时支持推理和训练；
- 能表达单设备和多设备；
- 能表达 prefill / decode loop / forward / backward / optimizer；
- 能把 profile 和预测都锚定到稳定的节点/边标识；
- 不依赖 nnScaler 专用 IR 类型。


## 2. 顶层对象

建议将 IR 文档统一建模为 `GraphSpec`。

```json
{
  "graph_id": "string",
  "graph_kind": "inference|training|optimizer|mixed",
  "frontend": "torch.fx|torch.export|aot_autograd|eager_trace",
  "runtime_target": "torch_eager_v1",
  "model_info": {},
  "execution_config": {},
  "nodes": [],
  "edges": [],
  "regions": [],
  "loop_regions": [],
  "device_mesh": {},
  "metadata": {}
}
```


## 3. 顶层字段定义

### 3.1 `graph_id`

- 全局唯一字符串；
- 用于报告、数据库回写和验证对齐。

### 3.2 `graph_kind`

- `inference`
- `training`
- `optimizer`
- `mixed`

### 3.3 `frontend`

记录图来源，建议值：

- `torch.fx`
- `torch.export`
- `aot_autograd`
- `eager_trace`

### 3.4 `runtime_target`

运行时假设，不是设备类型。v1 默认：

- `torch_eager_v1`


## 4. 节点模式

每个节点建模为 `NodeSpec`。

```json
{
  "node_id": "n123",
  "op_type": "aten.mm",
  "op_family": "gemm",
  "phase": "prefill|decode|forward|backward|optimizer",
  "module_scope": ["model", "layers.0", "self_attn"],
  "region_id": "r_attn_0",
  "inputs": ["t1", "t2"],
  "outputs": ["t3"],
  "attributes": {},
  "shape_signature": {},
  "dtype_signature": {},
  "layout_signature": {},
  "execution_annotation": {},
  "distribution_annotation": {},
  "cost_hint": {},
  "source_location": {},
  "metadata": {}
}
```

### 4.1 核心字段

- `node_id`
  - 稳定 ID；
- `op_type`
  - 原始算子类型，如 `aten.addmm`、`aten._scaled_dot_product_flash_attention`；
- `op_family`
  - 归一化族名，如 `gemm`、`attention`、`norm`、`pointwise`、`collective`；
- `phase`
  - 运行阶段标签；
- `module_scope`
  - 模块层次路径；
- `region_id`
  - 所属子图或逻辑 region。

### 4.2 `attributes`

保留原始算子关键参数，例如：

- `transposed_a`
- `transposed_b`
- `causal`
- `dropout_p`
- `activation_fn`
- `has_bias`

### 4.3 `shape_signature`

建议统一保存可用于数据库查询的归一化形状摘要，而不是仅保存原始输入列表。

示例：

```json
{
  "batch": 4,
  "seq": 2048,
  "hidden": 4096,
  "num_heads": 32,
  "num_kv_heads": 8,
  "head_dim": 128,
  "m": 8192,
  "n": 4096,
  "k": 4096
}
```

### 4.4 `dtype_signature`

建议区分：

- `input_dtype`
- `weight_dtype`
- `accum_dtype`
- `output_dtype`
- `kv_cache_dtype`

### 4.5 `layout_signature`

v1 可最小支持：

- `contiguous`: true/false
- `memory_format`: `contiguous|channels_last|unknown`
- `stride_hint`

### 4.6 `execution_annotation`

用于运行时组合：

- `stream_id`
- `launch_blocking`
- `is_fused`
- `is_recompute`
- `requires_grad`
- `checkpoint_group_id`
- `can_overlap_comm`

### 4.7 `distribution_annotation`

借鉴 nnScaler/nnScaler-M 的 partition 描述思路，但改成更通用的 schema：

```json
{
  "placement": [0, 1, 2, 3],
  "mesh_axes": ["dp", "tp"],
  "sharding": {
    "input": [{"tensor_id": "t1", "axis": "hidden", "parts": 4, "mode": "shard"}],
    "output": [{"tensor_id": "t3", "axis": "hidden", "parts": 4, "mode": "shard"}],
    "param": [{"tensor_id": "w1", "axis": "hidden", "parts": 4, "mode": "shard"}]
  },
  "collective_requirement": ["all_reduce"]
}
```


## 5. 张量模式

每个张量建模为 `TensorSpec`。张量可以内联在节点里，也可以在顶层单独建表。

```json
{
  "tensor_id": "t3",
  "shape": [4, 2048, 4096],
  "dtype": "bf16",
  "role": "activation|parameter|buffer|gradient|optimizer_state|kv_cache",
  "layout": {},
  "requires_grad": true,
  "producer": "n123",
  "consumers": ["n124", "n125"],
  "liveness": {},
  "distribution": {},
  "metadata": {}
}
```

### 5.1 `role`

必须显式区分：

- `activation`
- `parameter`
- `buffer`
- `gradient`
- `optimizer_state`
- `kv_cache`

这直接影响 profile 归因和内存/通信建模。


## 6. 边模式

边建模为 `EdgeSpec`。

```json
{
  "edge_id": "e12",
  "src_node": "n123",
  "dst_node": "n124",
  "tensor_id": "t3",
  "edge_kind": "data|control|comm",
  "phase": "forward|backward|prefill|decode|optimizer",
  "distribution_change": {},
  "comm_annotation": {},
  "metadata": {}
}
```

### 6.1 `edge_kind`

- `data`: 普通数据依赖；
- `control`: 调度依赖；
- `comm`: 显式通信边。

### 6.2 `comm_annotation`

若边是通信边，必须补充：

- `primitive`: `all_reduce|all_gather|reduce_scatter|broadcast|p2p|all_to_all`
- `payload_bytes`
- `src_mesh`
- `dst_mesh`
- `topology_hint`


## 7. Region 模式

为支持解释性输出和 decode loop 复用，IR 必须支持 region。

```json
{
  "region_id": "r_attn_0",
  "region_type": "attention|mlp|embedding|norm|optimizer|prefill|decode_step|pipeline_stage",
  "name": "layers.0.self_attn",
  "node_ids": ["n10", "n11", "n12"],
  "parent_region_id": "r_block_0",
  "metadata": {}
}
```

推荐最小 `region_type`：

- `prefill`
- `decode_step`
- `forward`
- `backward`
- `optimizer`
- `attention`
- `mlp`
- `pipeline_stage`


## 8. Loop region 模式

为表达 decode token loop，单独定义 `LoopRegionSpec`。

```json
{
  "loop_region_id": "loop_decode",
  "body_region_id": "r_decode_step",
  "loop_kind": "token_decode",
  "trip_count_expr": "max(output_tokens - 1, 0)",
  "loop_carried_tensors": ["kv_cache_0", "kv_cache_1"],
  "state_update_nodes": ["n901", "n902"],
  "metadata": {}
}
```

这部分是 AIConfigurator 风格 phase 拆解与 Torch eager 图建模之间的关键桥梁。


## 9. 设备与 mesh 模式

建议统一使用 `DeviceMeshSpec`：

```json
{
  "mesh_id": "mesh0",
  "devices": [0, 1, 2, 3],
  "axes": [
    {"name": "dp", "size": 2},
    {"name": "tp", "size": 2}
  ],
  "topology": {
    "intra_node": "nvlink",
    "inter_node": "infiniband"
  }
}
```


## 10. 执行配置模式

`execution_config` 建议至少含：

- `batch_size`
- `micro_batch_size`
- `seq_len`
- `output_len`
- `dtype`
- `gradient_accumulation_steps`
- `use_activation_checkpointing`
- `optimizer_type`
- `zero_stage`
- `tp_size`
- `dp_size`
- `pp_size`


## 11. 最小不变量

IR 必须满足以下不变量：

- 每个 `tensor_id` 只能有一个 producer；
- 每个 `node_id` 只能属于一个 phase；
- 每个 `node_id` 至少属于一个 region；
- 若存在 `comm` 边，则必须有 `payload_bytes`；
- 若 `graph_kind == training`，则至少要能区分 `forward`、`backward`；
- 若存在 decode loop，则必须显式有 `loop_region`。


## 12. v1 推荐的 IR 生成顺序

1. 从 `torch.fx` / `torch.export` / eager trace 采集原始图；
2. 标注张量 shape、dtype、role；
3. 划分 phase；
4. 识别 region；
5. 注入 distributed metadata；
6. 生成数据库查询键；
7. 生成解释性输出所需的 scope / region / stage 索引。


## 13. 与参考仓库的关系

- 借鉴 `nnscaler-M/nnscaler/autodist/descs.py` 对 partition/plan 输出结构化表达的思路；
- 借鉴 AIConfigurator 中按 phase 与 per-op breakdown 组织输出的做法；
- 不直接绑定 `IRDimops`、`NodePartitionDesc` 或某一后端专用 op taxonomy。
