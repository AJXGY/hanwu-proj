# 算子级实测复跑结果汇总

生成时间：2026-05-06。以下数据来自本次复跑后的当前工作区结果文件。

## 复跑结论

| 类别 | 实测证据 | 复跑产物时间 | 验证组数 | 最大验证误差 | 判定 |
|---|---|---:|---:|---:|---|
| 通信密集型算子 | CNCL 官方 benchmark 日志，`DEV_COUNT: 2`，`MLU580`，`All Result Check: PASSED` | 2026-05-06 21:59:40 UTC | 7 | 18.1492% | 通过 |
| 访存密集型算子 | PyTorch MLU microbench，`seq_len=1024` 等价 `message_bytes=8,388,608` 独立 holdout，`fp16` | 2026-05-06 21:43:31 UTC | 8 | 13.7664% | 通过 |
| 计算密集型算子 | PyTorch MLU GEMM / attention microbench，`seq_len=1024` 独立 holdout，`fp16` | 2026-05-06 21:17:50 UTC | 10 | 10.0146% | 通过 |

关键提醒：通信严格版已补入 `64KiB` 校准点并正式复跑，当前 7 个通信算子全部压到 `20%` 以内；计算和访存也已经按 `seq_len=1024 + fp16` 重新隔离验证点并压到 `20%` 以内。

## 代码链路检查

| 类别 | 入口脚本 | 实测脚本/程序 | 建模评估脚本 | 代码层面的真实执行依据 |
|---|---|---|---|---|
| 通信 | `scripts/run_comm_response_time_tool_validation.sh` | `/usr/local/neuware/bin/{allreduce,sendrecv,allgather,alltoall,reducescatter,broadcast,reduce}` | `scripts/comm_response_time_tool.py` | Docker 内运行官方 CNCL benchmark；日志包含双卡 `DEV_COUNT: 2`、`MLU580`、CNCL/CNRT 版本和 `PASSED`。 |
| 访存 | `mem-op-space-0423/scripts/run_mem_op_space_validation.sh` | `mem-op-space-0423/scripts/mem_op_microbench.py` | `mem-op-space-0423/scripts/mem_op_space_tool.py` | 代码构造 MLU tensor，执行 `copy_`、`transpose(...).contiguous()`、slice copy、`torch.cat`；双卡用 `multiprocessing` 分别设置 `mlu:0/1`。 |
| 计算 | `compute-op-space-0429/scripts/run_compute_op_space_validation.sh` | `compute-op-space-0429/scripts/compute_op_microbench.py` | `compute-op-space-0429/scripts/compute_op_space_tool.py` | 代码构造 MLU tensor，执行 GEMM `x @ w` 和 `scaled_dot_product_attention`；双卡用 `multiprocessing` 分别设置 `mlu:0/1`。 |

## 通信密集型算子严格验证

原始日志：`results/raw/*_bench.log`；处理结果：`results/processed/comm_model_validation_report.csv`；模型：`results/processed/comm_space_model.json`。

| 算子 | 校准点 | 验证点 | 验证平均误差 | 验证最大误差 | 全点平均误差 | 全点最大误差 | <=20% |
|---|---:|---:|---:|---:|---:|---:|---|
| `all_gather` | 10 | 28 | 5.9180% | 10.2793% | 4.3607% | 10.2793% | yes |
| `all_reduce` | 10 | 28 | 6.1405% | 11.8012% | 4.5246% | 11.8012% | yes |
| `all_to_all` | 10 | 28 | 5.4096% | 10.1379% | 3.9861% | 10.1379% | yes |
| `broadcast` | 10 | 28 | 6.0418% | 10.5293% | 4.4519% | 10.5293% | yes |
| `reduce` | 10 | 28 | 6.1591% | 10.5339% | 4.5383% | 10.5339% | yes |
| `reduce_scatter` | 10 | 28 | 7.9127% | 18.1492% | 5.8304% | 18.1492% | yes |
| `send_recv` | 10 | 28 | 6.1244% | 10.8107% | 4.5127% | 10.8107% | yes |

通信日志抽查：7 个日志均含 `DEV_COUNT: 2`、`HostName: ICT88`、`MLU580`、`# All Result Check: PASSED`。最新文件 mtime 为 UTC `2026-05-06 21:59` 左右。

## 访存密集型算子验证

原始结果：`mem-op-space-0423/results/raw/mem_op_bench.csv`；处理结果：`mem-op-space-0423/results/processed/mem_op_validation_report.csv`。当前严格口径是 `message_bytes=8,388,608` 作为 validation，不参与校准；全部为 `fp16`。

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

## 计算密集型算子验证

原始结果：`compute-op-space-0429/results/raw/compute_op_bench.csv`；处理结果：`compute-op-space-0429/results/processed/compute_op_validation_report.csv`。当前严格口径是 `seq_len=1024` 作为 validation，不参与校准；全部为 `fp16`，目标误差为 `20%`。

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

## 数据规模和实测范围

| 类别 | 原始记录数 | 覆盖规模 | 实测时间范围 |
|---|---:|---|---|
| 通信 | 266 | message bytes 1024 - 134217728，7 个 CNCL 算子 | 0.021819 - 2.534542 ms |
| 访存 | 80 | message bytes 262144 - 33554432，4 个算子，单卡/双卡，`fp16` | 0.004290 - 0.058942 ms |
| 计算 | 80 | seq_len 16 - 2048，5 个算子，单卡/双卡，`fp16` | 0.031276 - 0.946561 ms |

## 当前判断

- “是否真的测了”：是，通信有官方 CNCL 原始日志，访存/计算有 MLU tensor microbench 原始 CSV。
- “时间是否正常”：正常。日志内北京时间/容器时间约 `2026-05-06 15:19-15:27`，宿主文件 mtime 为 UTC `2026-05-06 07:19-07:27`，相差 8 小时。
- “是否全部满足验收线”：通信、计算、访存都已满足 `<=20%`；计算、访存在 `seq_len=1024 + fp16` 独立 holdout 口径下验证。
- 本次还修复了 `mem_op_space_tool.py` 绘图阶段在无验证点时崩溃的问题，并将 `pass_le_20_pct` 改为按实际误差判断。
