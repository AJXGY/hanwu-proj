# mem-op-space-0423

这个目录对应测试项：

- `CAMBRICON-MEM-OP-SPACE-TEST`
- 寒武纪架构访存密集型算子空间维度建模测试

当前目录用于单独验证 `Llama3.1-8B` 典型访存密集型算子在寒武纪 MLU 上的空间维度建模准确性，不与现有通信密集型任务混放。

## 覆盖算子

- `data_copy`
- `reshape_transpose`
- `slice_copy`
- `concat`

这些算子均使用 `Llama3.1-8B` 风格的 `hidden_size=4096`、`bf16` 激活张量形状构造微基准。

## 目录结构

- `scripts/mem_op_microbench.py`
  - 访存密集型算子微基准
- `scripts/mem_op_space_tool.py`
  - 算子级空间维度模型构建、单点预测与严格版评估
- `scripts/run_mem_op_space_validation.sh`
  - 一键执行 Docker benchmark、建模、误差统计和图表生成
- `results/raw/`
  - 原始 benchmark CSV
- `results/processed/`
  - 模型文件、误差统计表、摘要
- `figure/`
  - 图表输出
- `0423任务.md`
  - 当前测试项说明
- `0423进展.md`
  - 当前进展与结论

## 运行方式

```bash
cd /home/o_mabin/hanwu-proj/projects/operators/memory
bash scripts/run_mem_op_space_validation.sh
```

## 输出

- `results/raw/mem_op_bench.csv`
- `results/processed/mem_op_space_model.json`
- `results/processed/mem_op_validation_points.csv`
- `results/processed/mem_op_validation_report.csv`
- `figure/mem_op_validation_overview.png`
- `figure/strict/*.png`

## 当前结论

当前任务已完成。

最终验证结果见：

- [mem_op_validation_report.csv](/home/o_mabin/hanwu-proj/projects/operators/memory/results/processed/mem_op_validation_report.csv)
- [mem_op_validation_overview.png](/home/o_mabin/hanwu-proj/projects/operators/memory/figure/mem_op_validation_overview.png)

当前 8 组 `operator x scale` 最大验证误差均不超过 `20%`：

- `concat / single_card`: `1.2970%`
- `concat / single_node_dual_card`: `18.2609%`
- `data_copy / single_card`: `16.5155%`
- `data_copy / single_node_dual_card`: `0.1018%`
- `reshape_transpose / single_card`: `2.1401%`
- `reshape_transpose / single_node_dual_card`: `2.1197%`
- `slice_copy / single_card`: `12.6911%`
- `slice_copy / single_node_dual_card`: `5.1608%`
