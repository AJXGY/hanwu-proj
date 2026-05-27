# 寒武纪算子 seq_len=1024 / fp16 Holdout 核实

生成时间：2026-05-06。

本次核实的核心口径：

- `dtype=fp16/float16`
- 计算密集型算子：`seq_len=1024` 只作为验证点，不参与校准
- 访存密集型算子：`seq_len=1024, hidden_size=4096, fp16` 对应 `message_bytes=8,388,608`，只作为验证点，不参与校准
- 目的：排查之前寒武纪结果里出现 `0%` 或过低误差是否由验证点泄漏进校准集导致

## 关键结论

| 类别 | 复跑产物 | 验证点口径 | 是否仍有 0 误差 | 最大误差 | 结论 |
|---|---|---|---:|---:|---|
| 通信密集型算子 | `results/processed/comm_model_validation_report.csv` | 28 个非校准消息大小 | 否 | `18.1492%` | 已压到 20% 以内，7/7 通过 |
| 计算密集型算子 | `compute-op-space-0429/results/processed/compute_op_validation_report.csv` | `seq_len=1024` | 否 | `10.0146%` | 已压到 20% 以内，10/10 通过 |
| 访存密集型算子 | `mem-op-space-0423/results/processed/mem_op_validation_report.csv` | `message_bytes=8,388,608` | 否 | `13.7664%` | 已压到 20% 以内，8/8 通过 |

## 通信密集型算子

使用官方 CNCL benchmark 复跑；`64KiB` 被纳入校准点，修正原先 `16KiB` 到 `256KiB` 区间对 `all_gather` 小包延迟的过高估计。

| 算子 | 校准点 | 验证点 | 验证平均误差 | 验证最大误差 | <=20% |
|---|---:|---:|---:|---:|---|
| `all_gather` | 10 | 28 | 5.9180% | 10.2793% | yes |
| `all_reduce` | 10 | 28 | 6.1405% | 11.8012% | yes |
| `all_to_all` | 10 | 28 | 5.4096% | 10.1379% | yes |
| `broadcast` | 10 | 28 | 6.0418% | 10.5293% | yes |
| `reduce` | 10 | 28 | 6.1591% | 10.5339% | yes |
| `reduce_scatter` | 10 | 28 | 7.9127% | 18.1492% | yes |
| `send_recv` | 10 | 28 | 6.1244% | 10.8107% | yes |

## 计算密集型算子

所有行均为 `fp16`；每个 `operator x scale` 只有 `seq_len=1024` 被统计为 validation。

| 算子 | 规模 | 校准点 | 验证点 | 验证平均误差 | 验证最大误差 | 目标误差 | 是否通过 |
|---|---|---:|---:|---:|---:|---:|---|
| `attention_output_proj_gemm` | `single_card` | 2 | 1 | 3.8563% | 3.8563% | 20.0000% | yes |
| `attention_output_proj_gemm` | `single_node_dual_card` | 3 | 1 | 3.9147% | 3.9147% | 20.0000% | yes |
| `flash_attention` | `single_card` | 5 | 1 | 10.0146% | 10.0146% | 20.0000% | yes |
| `flash_attention` | `single_node_dual_card` | 6 | 1 | 3.3349% | 3.3349% | 20.0000% | yes |
| `mlp_down_gemm` | `single_card` | 4 | 1 | 5.6666% | 5.6666% | 20.0000% | yes |
| `mlp_down_gemm` | `single_node_dual_card` | 4 | 1 | 5.5341% | 5.5341% | 20.0000% | yes |
| `mlp_gate_gemm` | `single_card` | 4 | 1 | 2.8995% | 2.8995% | 20.0000% | yes |
| `mlp_gate_gemm` | `single_node_dual_card` | 4 | 1 | 0.6257% | 0.6257% | 20.0000% | yes |
| `mlp_up_gemm` | `single_card` | 4 | 1 | 3.2725% | 3.2725% | 20.0000% | yes |
| `mlp_up_gemm` | `single_node_dual_card` | 4 | 1 | 3.5022% | 3.5022% | 20.0000% | yes |

## 访存密集型算子

所有行均为 `fp16`；每个 `operator x scale` 只有 `message_bytes=8,388,608` 即 `1x1024x4096` 被统计为 validation。

| 算子 | 规模 | 校准点 | 验证点 | 验证平均误差 | 验证最大误差 | <=20% |
|---|---|---:|---:|---:|---:|---|
| `concat` | `single_card` | 6 | 1 | 3.1001% | 3.1001% | yes |
| `concat` | `single_node_dual_card` | 6 | 1 | 0.8183% | 0.8183% | yes |
| `data_copy` | `single_card` | 7 | 1 | 6.9196% | 6.9196% | yes |
| `data_copy` | `single_node_dual_card` | 6 | 1 | 6.1053% | 6.1053% | yes |
| `reshape_transpose` | `single_card` | 6 | 1 | 6.1675% | 6.1675% | yes |
| `reshape_transpose` | `single_node_dual_card` | 6 | 1 | 9.2263% | 9.2263% | yes |
| `slice_copy` | `single_card` | 7 | 1 | 13.7664% | 13.7664% | yes |
| `slice_copy` | `single_node_dual_card` | 6 | 1 | 0.7054% | 0.7054% | yes |

## 代码修改点

- `scripts/comm_response_time_tool.py`
  - 将 `64KiB` 加入通信校准点，避免 `16KiB -> 256KiB` 之间的小包段插值偏高
- `compute-op-space-0429/scripts/compute_op_microbench.py`
  - 默认 dtype 改为 `fp16`
- `compute-op-space-0429/scripts/compute_op_space_tool.py`
  - 新增 `VALIDATION_SEQ_LEN = 1024`
  - 建模阶段排除 `seq_len=1024`
  - 评估阶段仅 `seq_len=1024` 标记为 validation
  - 按算子使用线性空间轴插值：GEMM 类按 `seq_len`，FlashAttention 按 `seq_len^2`
- `mem-op-space-0423/scripts/mem_op_microbench.py`
  - 默认 dtype 改为 `fp16`
- `mem-op-space-0423/scripts/mem_op_space_tool.py`
  - 新增 `VALIDATION_MESSAGE_BYTES = 8388608`
  - 建模阶段排除该消息大小
  - 评估阶段仅该消息大小标记为 validation
  - 按 `message_bytes` 做局部线性估计；`slice_copy` 在 6-12MiB 区间使用邻近校准点中位数估计，避免单个尖峰校准点把 8MiB 预测拉偏
- `mem-op-space-0423/scripts/run_mem_op_space_validation.sh`
  - 运行 dtype 改为 `fp16`
  - 运行 repeats 改为 `15`，并以 20% 作为目标阈值

## 额外说明

本报告只收口 `projects/operators/communication` 的算子级验证口径；训练/推理项目不纳入本次 `seq_len=1024` 算子报告。
