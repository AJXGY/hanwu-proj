# hanwu-proj

这个仓库统一整理异构 AI 测试与建模项目。当前以寒武纪为主，后续摩尔线程等平台按同一目录和脚本命名接入。

结构规范见 [docs/repo_structure.md](/home/o_mabin/hanwu-proj/docs/repo_structure.md)。

## 项目总览

- [projects/operators/communication](/home/o_mabin/hanwu-proj/projects/operators/communication)：通信算子建模与验证。
- [projects/operators/memory](/home/o_mabin/hanwu-proj/projects/operators/memory)：存储类算子建模与验证。
- [projects/operators/compute](/home/o_mabin/hanwu-proj/projects/operators/compute)：计算类算子建模与验证。
- [projects/training/time-modeling](/home/o_mabin/hanwu-proj/projects/training/time-modeling)：训练时间建模、训练 smoke、训练 Dashboard。
- [projects/inference/time-modeling](/home/o_mabin/hanwu-proj/projects/inference/time-modeling)：推理时间建模、推理 smoke、推理 Dashboard。
- [projects/indicators/1.2-runtime-validation](/home/o_mabin/hanwu-proj/projects/indicators/1.2-runtime-validation)：指标 1.2 运行验证。
- [projects/indicators/1.3-execution-model](/home/o_mabin/hanwu-proj/projects/indicators/1.3-execution-model)：指标 1.3 处理模型输出验证。

历史汇总文档保存在 [docs/history](/home/o_mabin/hanwu-proj/docs/history)。

## 统一入口

推荐从仓库根目录运行：

```bash
bash scripts/run_inference_cambricon_dashboard.sh
bash scripts/run_training_cambricon_dashboard.sh
bash scripts/run_indicator_1_2_dashboard.sh
bash scripts/run_indicator_1_3_dashboard.sh
```

常用验证脚本：

```bash
bash scripts/run_operator_comm_cambricon.sh
bash scripts/run_operator_mem_cambricon.sh
bash scripts/run_operator_compute_cambricon.sh
bash scripts/run_training_single_cambricon.sh
bash scripts/run_training_multi_cambricon.sh
bash scripts/run_inference_single_cambricon.sh
bash scripts/run_inference_multi_cambricon.sh
```

训练和推理统一走 TP 路线。历史 PP 脚本不作为标准验证入口。

清理本地缓存和临时产物：

```bash
bash scripts/clean_workspace.sh
```

## Dashboard 端口

- 推理时间建模：`http://127.0.0.1:8123`
- 训练时间建模：`http://127.0.0.1:8234`
- 指标 1.2：`http://127.0.0.1:8242`
- 指标 1.3：`http://127.0.0.1:8253`

## 共享建议

建议提交源代码、README、任务/进展/ppt 文档、`scripts/`、`ui/`、小型配置和必要样例数据。

默认不提交 `reports/`、`validation_reports/`、`dashboard_runs/`、`tmp_*`、`__pycache__/`、日志、checkpoint 和大型中间结果。
