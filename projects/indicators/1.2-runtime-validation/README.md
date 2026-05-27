# Indicator 1.2 Runtime Validation

这个目录面向指标 `1.2` 的两项寒武纪异构适配运行测试：

- `5.1.3 寒武纪架构上推理任务运行测试`
- `5.1.4 寒武纪架构上训练任务运行测试`

这里的目标不是时间预测，而是直接验证：

- 单卡是否能正常启动、执行、结束
- 单机双卡是否能正常启动、执行、结束
- 推理能产出结果
- 训练能产出损失曲线与轻量 checkpoint 验证产物

## 目录说明

- [dashboard.py](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.2-runtime-validation/dashboard.py): 统一 Dashboard 后端
- [ui/index.html](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.2-runtime-validation/ui/index.html): Dashboard 页面
- [infer_task_runner.py](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.2-runtime-validation/infer_task_runner.py): 推理运行验证包装脚本
- [train_task_runner.py](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.2-runtime-validation/train_task_runner.py): 训练运行验证脚本
- [scripts/run_dashboard.sh](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.2-runtime-validation/scripts/run_dashboard.sh): Dashboard 启动脚本
- `reports/runs/`: 运行结果目录

## Dashboard

启动：

```bash
cd /home/o_mabin/hanwuji-clj-proj/projects/indicators/1.2-runtime-validation
bash scripts/run_dashboard.sh
```

打开：

```text
http://127.0.0.1:8242
```

## 运行说明

Dashboard 预置了四个按钮：

- `推理单卡`
- `推理双卡`
- `训练单卡`
- `训练双卡`

默认模型目录：

```text
/home/o_mabin/LLM/models/Llama-3.1-8B
```

默认 Docker 镜像：

```text
cambricon-base/pytorch:v25.10.0-torch2.7.1-torchmlu1.29.1-ubuntu22.04-py310
```

## 输出产物

推理运行输出：

- `summary.json`
- `report.json`
- `report.md`
- `stdout.log`
- `stderr.log`

训练运行输出：

- `summary.json`
- `loss_history.json`
- `loss_curve.svg`
- `checkpoint/*.pt`

训练这里保存的是`轻量 checkpoint 验证产物`，用于证明训练完成后保存链路可用，同时避免把完整 8B 权重再次落盘。

## 当前已验证结果

推理：

- 单卡成功：
  - [c1b1b382/summary.json](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.2-runtime-validation/reports/runs/inference/c1b1b382/summary.json)
- 双卡成功：
  - [180d4000/summary.json](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.2-runtime-validation/reports/runs/inference/180d4000/summary.json)

训练：

- 单卡成功：
  - [8b0e444b/summary.json](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.2-runtime-validation/reports/runs/training_local/8b0e444b/summary.json)
  - [8b0e444b/loss_curve.svg](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.2-runtime-validation/reports/runs/training_local/8b0e444b/loss_curve.svg)
- 双卡成功：
  - [166410aa/summary.json](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.2-runtime-validation/reports/runs/training_local/166410aa/summary.json)
  - [166410aa/loss_curve.svg](/home/o_mabin/hanwuji-clj-proj/projects/indicators/1.2-runtime-validation/reports/runs/training_local/166410aa/loss_curve.svg)
