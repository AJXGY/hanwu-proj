# Indicator 1.3 Execution Model

这个目录面向指标 `1.3` 的两项处理模型输出验证：

- `5.1.9 寒武纪架构上训练任务处理模型输出测试`
- `5.1.10 寒武纪架构上人工智能推理任务处理模型输出测试`

这里的目标不是“把任务跑完”，而是：

- 输入 `Llama3.1-8B` 的训练/推理配置
- 输出执行模型 JSON
- 输出 CPU 与 NPU 的任务分配关系
- 输出多卡划分关系与并行方式说明
- 输出 microbatch 执行逻辑
- 输出 DAG 图与可视化页面

## 主要文件

- [dashboard.py](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.3-execution-model/dashboard.py): 统一 Dashboard 后端
- [ui/index.html](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.3-execution-model/ui/index.html): Dashboard 页面
- [infer_model_builder.py](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.3-execution-model/infer_model_builder.py): 推理处理模型构建脚本
- [train_model_builder.py](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.3-execution-model/train_model_builder.py): 训练处理模型构建脚本
- [scripts/run_dashboard.sh](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.3-execution-model/scripts/run_dashboard.sh): Dashboard 启动脚本

## Dashboard

启动：

```bash
cd /home/o_mabin/hanwuji-clj-proj/projects/indicators/1.3-execution-model
bash scripts/run_dashboard.sh
```

打开：

```text
http://127.0.0.1:8253
```

## 输出产物

推理：

- `execution_model.json`
- `execution_dag.json`
- `execution_dag.svg`
- `graph_viz/`
- `index.html`
- `summary.json`

训练：

- `execution_model.json`
- `execution_dag.json`
- `execution_dag.svg`
- `index.html`
- `summary.json`

## 预设

页面里提供四个预设：

- `推理单卡`
- `推理双卡`
- `训练单卡`
- `训练双卡`

## 已完成结果

当前已经基于真实 `Llama3.1-8B` 默认模型路径
`/home/o_mabin/LLM/models/Llama-3.1-8B`
生成 4 组处理模型产物：

- 推理单卡：
  [01e5bf38](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.3-execution-model/reports/models/inference/01e5bf38)
- 推理双卡：
  [fe2c9aa5](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.3-execution-model/reports/models/inference/fe2c9aa5)
- 训练单卡：
  [645d804a](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.3-execution-model/reports/models/training/645d804a)
- 训练双卡：
  [8b994f49](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.3-execution-model/reports/models/training/8b994f49)

其中可直接查看的关键文件包括：

- 推理执行模型：
  [execution_model.json](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.3-execution-model/reports/models/inference/fe2c9aa5/execution_model.json)
- 推理 DAG 图：
  [execution_dag.svg](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.3-execution-model/reports/models/inference/fe2c9aa5/execution_dag.svg)
- 推理 graph 可视化：
  [graph_viz/index.html](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.3-execution-model/reports/models/inference/fe2c9aa5/graph_viz/index.html)
- 训练执行模型：
  [execution_model.json](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.3-execution-model/reports/models/training/8b994f49/execution_model.json)
- 训练 DAG 图：
  [execution_dag.svg](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.3-execution-model/reports/models/training/8b994f49/execution_dag.svg)

## 验证结论

当前 `1.3.md` 中两项任务都已覆盖：

- `5.1.9 寒武纪架构上训练任务处理模型输出测试`
- `5.1.10 寒武纪架构上人工智能推理任务处理模型输出测试`

实际输出已覆盖以下要求：

- CPU 与 NPU 的任务分配方式
- 多卡划分关系
- 并行方式说明
- microbatch 执行逻辑
- DAG 图与可视化产物

推理双卡产物中可看到：

- `parallel_mode = tp`
- `world_size = 2`
- `two-card tensor-parallel sharding with collective synchronization`

训练双卡产物中可看到：

- `parallel_mode = pipeline_parallel`
- `pp_size = 2`
- `2 microbatches flow across 2 pipeline stages`
- `layer_range [0, 15]` 与 `[16, 31]` 的阶段划分
