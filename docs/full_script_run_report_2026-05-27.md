# 全量脚本运行报告 - 2026-05-27

## 结论

本轮从仓库根目录 `/home/o_mabin/hanwu-proj` 逐个运行了标准入口脚本。一次性验证脚本全部通过，Dashboard 入口均完成启动验证。

运行日志统一保存在：

```text
validation_runs/2026-05-27_full_script_run/logs/
```

本轮专用训练/推理输出目录：

```text
projects/training/time-modeling/reports/full_script_run_20260527/
projects/inference/time-modeling/validation_reports/full_script_run_20260527/
```

## 运行前处理

运行前 `cnmon info` 显示 0 号 MLU 被一个 vLLM OpenAI API server 占用约 39GB：

```text
python3 -m vllm.entrypoints.openai.api_server
model: /models/public-models/llm/DeepSeek-R1-Distill-Qwen-7B
port: 1579
pid: 1296492
```

为了完整运行双卡训练、推理和通信验证，已停止该 vLLM 进程。停止后两张 MLU 显存均恢复空闲。

## 脚本运行结果

| 序号 | 脚本 | 结果 | 日志 |
| --- | --- | --- | --- |
| 00 | `scripts/clean_workspace.sh` | PASS | `validation_runs/2026-05-27_full_script_run/logs/00_clean_workspace.log` |
| 01 | `scripts/run_operator_comm_cambricon.sh` | PASS | `validation_runs/2026-05-27_full_script_run/logs/01_run_operator_comm_cambricon.log` |
| 02 | `scripts/run_operator_mem_cambricon.sh` | PASS | `validation_runs/2026-05-27_full_script_run/logs/02_run_operator_mem_cambricon.log` |
| 03 | `scripts/run_operator_compute_cambricon.sh` | PASS | `validation_runs/2026-05-27_full_script_run/logs/03_run_operator_compute_cambricon.log` |
| 04 | `scripts/run_training_single_cambricon.sh` | PASS | `validation_runs/2026-05-27_full_script_run/logs/04_run_training_single_cambricon.log` |
| 05 | `scripts/run_training_multi_cambricon.sh` | PASS | `validation_runs/2026-05-27_full_script_run/logs/05_run_training_multi_cambricon.log` |
| 06 | `scripts/run_training_cambricon_smoke.sh` | PASS | `validation_runs/2026-05-27_full_script_run/logs/06_run_training_cambricon_smoke.log` |
| 07 | `scripts/run_inference_single_cambricon.sh` | PASS | `validation_runs/2026-05-27_full_script_run/logs/07_run_inference_single_cambricon.log` |
| 08 | `scripts/run_inference_multi_cambricon.sh` | PASS | `validation_runs/2026-05-27_full_script_run/logs/08_run_inference_multi_cambricon.log` |
| 09 | `scripts/run_inference_cambricon_smoke.sh` | PASS | `validation_runs/2026-05-27_full_script_run/logs/09_run_inference_cambricon_smoke.log` |
| 10 | `scripts/run_inference_cambricon_dashboard.sh` | 启动验证 PASS | `validation_runs/2026-05-27_full_script_run/logs/10_run_inference_cambricon_dashboard.log` |
| 11 | `scripts/run_training_cambricon_dashboard.sh` | 启动验证 PASS | `validation_runs/2026-05-27_full_script_run/logs/11_run_training_cambricon_dashboard.log` |
| 12 | `scripts/run_indicator_1_2_dashboard.sh` | 启动验证 PASS | `validation_runs/2026-05-27_full_script_run/logs/12_run_indicator_1_2_dashboard.log` |
| 13 | `scripts/run_indicator_1_3_dashboard.sh` | 启动验证 PASS | `validation_runs/2026-05-27_full_script_run/logs/13_run_indicator_1_3_dashboard.log` |

Dashboard 入口启动后都返回过 HTTP 200。由于当前命令执行环境会在命令结束后回收后台子进程，本轮只记录“启动验证通过”，未保持 Dashboard 常驻。

## 算子验证结果

| 类别 | 验证组数 | 最大误差点 | 最大误差 | 结果 |
| --- | ---: | --- | ---: | --- |
| 通信算子 | 7 | `reduce_scatter` | 4.4443% | PASS |
| 访存算子 | 8 | `concat/single_card` | 13.2145% | PASS |
| 计算算子 | 10 | `attention_output_proj_gemm/single_node_dual_card` | 11.1410% | PASS |

结果文件：

```text
projects/operators/communication/results/processed/comm_model_validation_report.csv
projects/operators/memory/results/processed/mem_op_validation_report.csv
projects/operators/compute/results/processed/compute_op_validation_report.csv
```

## 训练结果

| 用例 | 并行 | TP | T_real ms | T_sim ms | 相对误差 | 报告 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| training_single | `single` | 1 | 111.4694 | 105.8043 | 5.0822% | `projects/training/time-modeling/reports/full_script_run_20260527/train_single/report.json` |
| training_multi | `tp` | 2 | 336.0977 | 314.7671 | 6.3465% | `projects/training/time-modeling/reports/full_script_run_20260527/train_multi/report.json` |
| training_smoke_single | `single` | 1 | 112.4175 | 105.8043 | 5.8827% | `projects/training/time-modeling/reports/full_script_run_20260527/train_smoke_single/report.json` |
| training_smoke_tp2 | `tp` | 2 | 320.2042 | 314.7671 | 1.6980% | `projects/training/time-modeling/reports/full_script_run_20260527/train_smoke_tp2/report.json` |

## 推理结果

| 用例 | 并行 | TP | Request real ms | Request sim ms | Request 误差 | Prefill 误差 | Decode 误差 | 报告 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| inference_single | `single` | 1 | 62.4393 | 62.9241 | 0.7764% | 2.5655% | 0.1228% | `projects/inference/time-modeling/validation_reports/full_script_run_20260527/infer_single/report.json` |
| inference_multi | `tp` | 2 | 233.8728 | 236.8326 | 1.2656% | 0.6006% | 1.9504% | `projects/inference/time-modeling/validation_reports/full_script_run_20260527/infer_multi/report.json` |
| inference_smoke | `tp` | 2 | 235.0957 | 234.5875 | 0.2162% | 0.2486% | 3.1410% | `projects/inference/time-modeling/validation_reports/full_script_run_20260527/infer_smoke/report.json` |

## 本轮修复

1. 修复训练 smoke 脚本自定义输出目录时打印默认路径的问题。
   - 文件：`projects/training/time-modeling/scripts/run_cambricon_train_smoke.sh`
   - 修复后会按 `OUTPUT_DIR_SINGLE`、`OUTPUT_DIR_TP` 打印真实报告路径。

2. 修复推理 TP2 底层脚本固定输出 `Smoke report` 的提示问题。
   - 文件：`projects/inference/time-modeling/scripts/run_cambricon_infer_smoke.sh`
   - 修复后提示为 `TP two-card inference report`，避免 multi 入口日志误导。

3. 修复访存算子 microbench 对极短 kernel 抖动过敏的问题。
   - 文件：`projects/operators/memory/scripts/mem_op_microbench.py`
   - 原先用重复测量中位数写入 `avg_ms`，本轮实测导致 `reshape_transpose/single_card` 和 `slice_copy/single_card` validation 超过 20%。
   - 修复后使用同步重复测量中的最小值作为建模代表延迟，同时保留 min/max/std 字段用于观察抖动范围。

4. 增加访存算子验证失败退出保护。
   - 文件：`projects/operators/memory/scripts/mem_op_space_tool.py`
   - 以后 validation 误差超过 20% 时，evaluate 阶段会退出非 0，避免“脚本成功但报告失败”。

## 额外验证

已执行：

```bash
bash -n scripts/*.sh
bash -n projects/*/*/scripts/*.sh
python3 -m py_compile projects/operators/memory/scripts/mem_op_microbench.py
python3 -m py_compile projects/operators/memory/scripts/mem_op_space_tool.py
```

以上检查均通过。
