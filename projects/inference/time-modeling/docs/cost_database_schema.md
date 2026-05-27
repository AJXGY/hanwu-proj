# 代价数据库模式

源码对应关系见 `docs/source_traceability.md`。

## 1. 目标

代价数据库负责承载离线分析结果、通信 profile、修正系数和验证记录，使预测系统能够：

- 对单算子、模块、通信原语进行稳定查询；
- 区分 measured / interpolated / extrapolated / calibrated；
- 支持训练和推理的统一查询接口；
- 允许随硬件、软件版本、运行时模型扩展。


## 2. 分层结构

建议将数据库拆成四层：

1. `compute_profile`
   - 算子或模块级计算 profile；
2. `comm_profile`
   - 通信原语 profile；
3. `calibration_profile`
   - 修正层记录；
4. `validation_record`
   - 预测 vs 实测验证记录。

`validation_record` 不直接参与推理，但用于校准和回归比较。


## 3. 顶层元数据

每个数据库包必须带 `DatabaseManifest`：

```json
{
  "schema_version": "v1",
  "database_id": "a100_sxm_torch_eager_v1",
  "created_at": "2026-03-20T12:00:00Z",
  "producer": "torch-profiler-v1",
  "runtime_model": "torch_eager_v1",
  "hardware": {},
  "software": {},
  "units": {
    "time": "us",
    "memory": "bytes"
  }
}
```


## 4. 计算 profile 模式

### 4.1 记录类型

建议统一使用 `ComputeProfileRecord`。

```json
{
  "record_id": "cp_0001",
  "record_kind": "op|module",
  "target_kind": "inference|training|both",
  "runtime_model": "torch_eager_v1",
  "hardware_key": "a100_sxm_80g",
  "software_key": "torch_2.6_cuda_12.4",
  "query_key": {},
  "measurement": {},
  "memory": {},
  "statistics": {},
  "provenance": {},
  "evidence_level": "measured",
  "tags": []
}
```

### 4.2 `query_key`

这是最重要的索引部分，建议包含：

- `op_type`
- `op_family`
- `phase`
- `input_signature`
- `output_signature`
- `shape_signature`
- `dtype_signature`
- `layout_signature`
- `requires_grad`
- `is_recompute`
- `distributed_signature`

其中 `shape_signature` 应优先使用归一化命名字段，而非纯位置数组。

### 4.3 `measurement`

建议至少包含：

- `forward_time_us`
- `backward_time_us`
- `recompute_time_us`
- `runtime_overhead_time_us`
- `memory_stall_correction_time_us`

这些字段与时间语义规范一一对应。

### 4.4 `memory`

借鉴 nnScaler / nnScaler-M 的拆分思路，建议至少保存：

- `input_bytes`
- `param_bytes`
- `buffer_bytes`
- `saved_activation_bytes`
- `peak_infer_bytes`
- `peak_train_bytes`

### 4.5 `statistics`

建议至少包含：

- `warmup_runs`
- `measured_runs`
- `mean_us`
- `median_us`
- `std_us`
- `p95_us`
- `cov`

### 4.6 `provenance`

记录来源：

- `source_graph_id`
- `source_node_ids`
- `profiler_version`
- `command`
- `git_sha`
- `collector_host`


## 5. 通信 profile 模式

### 5.1 记录类型

建议统一使用 `CommProfileRecord`。

```json
{
  "record_id": "comm_0001",
  "runtime_model": "torch_eager_v1",
  "hardware_key": "a100_sxm_80g",
  "software_key": "nccl_2.21_cuda_12.4",
  "query_key": {},
  "curve": {},
  "statistics": {},
  "evidence_level": "measured",
  "tags": []
}
```

### 5.2 `query_key`

建议包含：

- `primitive`
- `world_size`
- `mesh_shape`
- `topology`
- `dtype`
- `transport`
- `intra_node`
- `inter_node`

### 5.3 `curve`

建议使用 piecewise 数据，而不是只保存单点：

```json
{
  "message_sizes_bytes": [1048576, 4194304, 16777216],
  "latency_us": [120.0, 240.0, 800.0],
  "bandwidth_gbps": [70.0, 130.0, 160.0],
  "fit_kind": "piecewise_linear"
}
```

这与 nnScaler / nnScaler-M 的通信 profile 查询方式兼容，但 schema 更显式。


## 6. 修正层模式

### 6.1 记录类型

建议使用 `CalibrationRecord`。

```json
{
  "record_id": "cal_0001",
  "runtime_model": "torch_eager_v1",
  "scope": "global|op_family|model_family|hardware|phase",
  "match_key": {},
  "correction_kind": "scale|bias|lookup|residual_model",
  "parameters": {},
  "derived_from": [],
  "validity": {}
}
```

### 6.2 建议支持的修正类型

- `global_scale`
- `global_bias_us`
- `op_family_scale`
- `seq_len_bucket_scale`
- `decode_loop_overhead_scale`
- `training_optimizer_bias_us`

v1 建议从简单的线性修正开始。


## 7. 验证记录模式

### 7.1 记录类型

建议使用 `ValidationRecord`。

```json
{
  "record_id": "val_0001",
  "benchmark_case_id": "llama3_8b_infer_bs1_isl2048_osl128",
  "runtime_model": "torch_eager_v1",
  "prediction": {},
  "measurement": {},
  "error": {},
  "artifacts": {}
}
```

### 7.2 `error`

建议至少包含：

- `abs_error_ms`
- `relative_error`
- `mape`
- `component_errors`
- `topk_bottleneck_rank_correlation`


## 8. 查询顺序

数据库查询必须采用确定性优先级：

1. 精确命中 measured record；
2. 精确命中 calibrated record；
3. 插值命中邻近 measured records；
4. 外推命中同 op family record；
5. 回退到 module-level record；
6. 回退到 analytic fallback。

每次查询都必须返回：

- 值；
- `evidence_level`；
- 匹配记录 ID；
- 是否插值/外推。


## 9. 建议的物理存储布局

v1 可以采用简单文件布局：

```text
database/
  manifest.json
  compute/
    op_family=gemm/*.jsonl
    op_family=attention/*.jsonl
    op_family=norm/*.jsonl
  comm/
    primitive=all_reduce/*.jsonl
    primitive=all_gather/*.jsonl
  calibration/
    *.jsonl
  validation/
    *.jsonl
```

若后续需要更强查询性能，可迁移到 SQLite / DuckDB / Parquet。


## 10. 最小主键建议

### 10.1 计算记录

推荐联合主键：

- `runtime_model`
- `hardware_key`
- `software_key`
- `op_type`
- `phase`
- `shape_signature_hash`
- `dtype_signature_hash`
- `distributed_signature_hash`

### 10.2 通信记录

推荐联合主键：

- `hardware_key`
- `software_key`
- `primitive`
- `world_size`
- `mesh_shape`
- `dtype`
- `topology`


## 11. 与参考仓库的关系

- 继承 nnScaler / nnScaler-M 将计算 profile 与通信 profile 分离的思想；
- 继承 `fw_span` / `bw_span` 和显式 memory buckets 的核心字段；
- 吸收 AIConfigurator 中“系统元数据 + 每类 op 数据文件 + 明确查询 API”的组织方式；
- 不继承 AIConfigurator 对特定 serving backend kernel taxonomy 的强绑定。
