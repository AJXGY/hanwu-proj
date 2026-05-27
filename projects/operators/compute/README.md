# compute-op-space-0429

这个目录对应测试项：

- `CAMBRICON-COMPUTE-OP-SPACE-TEST`
- 寒武纪架构计算密集型算子空间维度建模测试

当前目录用于单独验证 `Llama3.1-8B` 典型计算密集型算子在寒武纪 MLU 上的空间维度建模准确性。

## 覆盖算子

- `mlp_up_gemm`
- `mlp_gate_gemm`
- `mlp_down_gemm`
- `flash_attention`
- `attention_output_proj_gemm`

这些算子对应：

- `MLP Up GEMM`
- `MLP Gate GEMM`
- `MLP Down GEMM`
- `FlashAttention`
- `Attention Output Proj GEMM`

统一采用：

- `hidden_size = 4096`
- `intermediate_size = 14336`
- `num_heads = 32`
- `bf16`
- `single_card`
- `single_node_dual_card`

## 目录结构

- `scripts/compute_op_microbench.py`
  - 计算密集型算子微基准
- `scripts/compute_op_space_tool.py`
  - 算子级空间维度模型构建、单点预测与严格版评估
- `scripts/run_compute_op_space_validation.sh`
  - 一键执行 Docker benchmark、建模、误差统计和图表生成
- `results/raw/`
  - 原始 benchmark CSV
- `results/processed/`
  - 模型文件、误差统计表、摘要
- `figure/`
  - 图表输出
- `0429任务.md`
  - 当前测试项说明
- `0429进展.md`
  - 当前进展与结论

## 运行方式

```bash
cd /home/o_mabin/hanwu-proj/projects/operators/compute
bash scripts/run_compute_op_space_validation.sh
```

## 当前结果

当前目录已经完成新的 5 类代表性算子验证，结果位于：

- `results/raw/compute_op_bench.csv`
- `results/processed/compute_op_space_model.json`
- `results/processed/compute_op_validation_points.csv`
- `results/processed/compute_op_validation_report.csv`
- `figure/compute_op_validation_overview.png`
- `figure/strict/*.png`

当前结果如下：

- `attention_output_proj_gemm / single_card`: 最大误差 `3.8242%`
- `attention_output_proj_gemm / single_node_dual_card`: 最大误差 `2.1137%`
- `flash_attention / single_card`: 最大误差 `0.3791%`
- `flash_attention / single_node_dual_card`: 最大误差 `0.7548%`
- `mlp_down_gemm / single_card`: 最大误差 `8.5287%`
- `mlp_down_gemm / single_node_dual_card`: 最大误差 `8.6186%`
- `mlp_gate_gemm / single_card`: 最大误差 `9.6937%`
- `mlp_gate_gemm / single_node_dual_card`: 最大误差 `7.9163%`
- `mlp_up_gemm / single_card`: 最大误差 `9.8252%`
- `mlp_up_gemm / single_node_dual_card`: 最大误差 `8.2561%`

当前复线脚本默认按 `10%` 目标误差建模，当前 10 组结果全部满足 `<= 10%`。
