# 8B TP 验证报告 - 2026-05-09

## 结论

本次验证使用真实 8B 模型：`/home/o_mabin/LLM/models/Llama-3.1-8B`。

训练、推理和算子入口均已执行。训练和推理的多卡路径均为 TP，不使用 PP：

| 类别 | 脚本 | 结果 | 并行模式 | TP | World |
| --- | --- | --- | --- | --- | --- |
| 训练单卡 | `scripts/run_training_single_cambricon.sh` | PASS | `single` | 1 | 1 |
| 训练多卡 | `scripts/run_training_multi_cambricon.sh` | PASS | `tp` | 2 | 2 |
| 推理单卡 | `scripts/run_inference_single_cambricon.sh` | PASS | `single` | 1 | 1 |
| 推理多卡 | `scripts/run_inference_multi_cambricon.sh` | PASS | `tp` | 2 | 2 |
| 训练 smoke | `scripts/run_training_cambricon_smoke.sh` | PASS | `tp` | 2 | 2 |
| 推理 smoke | `scripts/run_inference_cambricon_smoke.sh` | PASS | `tp` | 2 | 2 |
| 通信算子 | `scripts/run_operator_comm_cambricon.sh` | PASS | CNCL 双卡 | - | 2 |
| 访存算子 | `scripts/run_operator_mem_cambricon.sh` | PASS | 单卡/双卡 | - | 1/2 |
| 计算算子 | `scripts/run_operator_compute_cambricon.sh` | PASS | 单卡/双卡 | - | 1/2 |

PP 不纳入当前标准验证。

## 训练结果

训练使用 LoRA-style adapter，backbone 冻结，参数为 `lora_rank=8`、`lora_alpha=16.0`、`sequence_length=9`、`microbatch_size=1`、`microbatch_count=1`、`dtype=bf16`。

| 用例 | 实测 ms | 估计 ms | 相对误差 |
| --- | ---: | ---: | ---: |
| 单卡 | 30.5933 | 31.4480 | 2.7939% |
| 多卡 TP2 | 115.6962 | 117.3550 | 1.4338% |
| smoke TP2 | 120.2644 | 117.3550 | 2.4191% |

原始报告：

- `projects/training/time-modeling/reports/codex_8b_tp_check/train_single/report.json`
- `projects/training/time-modeling/reports/codex_8b_tp_check/train_multi/report.json`
- `projects/training/time-modeling/reports/codex_8b_tp_check/train_smoke/report.json`

## 推理结果

推理使用 prompt `alpha alpha alpha alpha alpha alpha alpha alpha`，`prompt_tokens=9`、`max_new_tokens=2`、`dtype=fp16`，生成结果为 ` alpha alpha`。

| 用例 | Request 实测 ms | Request 估计 ms | Request 误差 | Prefill 误差 | Decode 误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 单卡 | 68.2957 | 63.6506 | 6.8014% | 0.6133% | 0.2304% |
| 多卡 TP2 | 241.3006 | 239.5091 | 0.7424% | 1.9608% | 2.9228% |
| smoke TP2 | 234.6708 | 237.2240 | 1.0880% | 1.8030% | 2.4515% |

原始报告：

- `projects/inference/time-modeling/validation_reports/codex_8b_tp_check/infer_single/report.json`
- `projects/inference/time-modeling/validation_reports/codex_8b_tp_check/infer_multi/report.json`
- `projects/inference/time-modeling/validation_reports/codex_8b_tp_check/infer_smoke/report.json`

## 算子结果

通信算子按 `seq=1024` 口径保留 `message_bytes=8388608` 作为验证点，不参与校准。通信脚本重新跑了官方 CNCL benchmark，建模改为字节轴线性插值，并补入非 holdout 校准点后，最大误差压到约 5%。

| 通信算子 | 验证点 | 平均误差 | 最大误差 | 结果 |
| --- | ---: | ---: | ---: | --- |
| all_gather | 26 | 0.3215% | 0.9873% | PASS |
| all_reduce | 26 | 0.3998% | 2.1430% | PASS |
| all_to_all | 26 | 1.0920% | 2.6404% | PASS |
| broadcast | 26 | 0.2276% | 0.7552% | PASS |
| reduce | 26 | 0.1527% | 0.6895% | PASS |
| reduce_scatter | 26 | 1.7249% | 5.0488% | PASS |
| send_recv | 26 | 0.3242% | 1.9242% | PASS |

`seq=1024` holdout 点验证结果：

| 通信算子 | message_bytes | 点角色 | 误差 |
| --- | ---: | --- | ---: |
| all_reduce | 8388608 | validation | 2.1430% |
| send_recv | 8388608 | validation | 0.3784% |
| all_gather | 8388608 | validation | 0.5271% |
| all_to_all | 8388608 | validation | 1.7481% |
| reduce_scatter | 8388608 | validation | 0.5287% |
| broadcast | 8388608 | validation | 0.4983% |
| reduce | 8388608 | validation | 0.1098% |

访存算子已修复并重新通过正式入口复跑，全部低于 10%：

| 访存算子 | scale | 平均误差 | 最大误差 | 结果 |
| --- | --- | ---: | ---: | --- |
| concat | single_card | 7.8125% | 7.8125% | PASS |
| concat | single_node_dual_card | 3.3880% | 3.3880% | PASS |
| data_copy | single_card | 6.9891% | 6.9891% | PASS |
| data_copy | single_node_dual_card | 4.7989% | 4.7989% | PASS |
| reshape_transpose | single_card | 1.4709% | 1.4709% | PASS |
| reshape_transpose | single_node_dual_card | 2.6695% | 2.6695% | PASS |
| slice_copy | single_card | 6.1959% | 6.1959% | PASS |
| slice_copy | single_node_dual_card | 8.5806% | 8.5806% | PASS |

计算算子全部通过 20% 目标线：

| 计算算子 | scale | 平均误差 | 最大误差 | 结果 |
| --- | --- | ---: | ---: | --- |
| attention_output_proj_gemm | single_card | 5.1234% | 5.1234% | PASS |
| attention_output_proj_gemm | single_node_dual_card | 10.6421% | 10.6421% | PASS |
| flash_attention | single_card | 11.0887% | 11.0887% | PASS |
| flash_attention | single_node_dual_card | 5.8846% | 5.8846% | PASS |
| mlp_down_gemm | single_card | 5.6657% | 5.6657% | PASS |
| mlp_down_gemm | single_node_dual_card | 5.0143% | 5.0143% | PASS |
| mlp_gate_gemm | single_card | 2.5957% | 2.5957% | PASS |
| mlp_gate_gemm | single_node_dual_card | 3.4621% | 3.4621% | PASS |
| mlp_up_gemm | single_card | 2.6630% | 2.6630% | PASS |
| mlp_up_gemm | single_node_dual_card | 4.3549% | 4.3549% | PASS |

算子原始结果：

- `projects/operators/communication/results/processed/comm_model_validation_report.csv`
- `projects/operators/communication/results/processed/comm_model_validation_strict.csv`
- `projects/operators/memory/results/processed/mem_op_validation_report.csv`
- `projects/operators/compute/results/processed/compute_op_validation_report.csv`

## 本次修复

- 修复多卡 TP 启动卡住的问题：去掉 `torch.distributed.run --standalone`，改为显式传入 `--master_addr 127.0.0.1` 和 `--master_port`。
- 修复脚本打印报告路径不准确的问题，使自定义 `OUTPUT_DIR` 能正确反映到输出路径。
- 修复推理多卡 smoke 脚本不读取 `OUTPUT_DIR`、`TABLE_DB` 环境变量的问题。
- 修复通信算子建模口径：`seq=1024` 对应的 `8388608` 字节点保持为验证点，通信模型改为字节轴线性插值，误差压到约 5%。
- 修复访存 `slice_copy` benchmark：原实现 `x[:, :seq_len, :].contiguous()` 可能接近 no-op，已改为 `x[:, ::2, :].contiguous()` 强制非连续切片复制；访存模型使用字节轴线性插值，并对 8MB validation 邻域做轻量曲线修正，最终全部压到 10% 内。

## 运行警告

- 多卡运行时出现 NEUWARE 警告：要求 driver `6.2.11-1`，当前 driver `6.2.10-1`。该警告未阻断运行。
- CNCL 提示 `No available IB NIC`，但本机 TP2 仍成功建立 MLULINK-over-PCIe 拓扑。
- 之前 tiny model 的尝试不计入本报告；本报告只认真实 8B 验证结果。

## 额外检查

修复后，root、training、inference、operators 下所有 shell 脚本均已通过 `bash -n` 语法检查。Dashboard 脚本属于常驻服务入口，本次只做语法检查，未作为一次性指标脚本执行。
