# 训练验证摘要

当前 `hanwu-proj` 的训练预测主线已经从历史 PP 实验切换为 TP：

- 单机单卡：`parallel_mode=single, tp_size=1`
- 单机多卡：`parallel_mode=tp, tp_size=2`
- 时间模型：`T_sim = microbatch_count * (forward + backward_compute + backward_comm) + optimizer_step`
- 参考方法：`/home/o_mabin/train-infer-estimation` 的阶段化训练估算语义
- 训练任务：`lora_vocab_adapter`，即 backbone backward + `hidden_size -> rank -> vocab_size` adapter + token-level vocab CE

历史 pipeline 实验文件和旧 profile 表仅作为归档材料保留；标准训练入口、Dashboard 和 smoke 验证不再调用历史路线。

当前验证入口：

```bash
SEQUENCE_LENGTH=8 MICROBATCH_COUNT=1 bash scripts/run_training_cambricon_smoke.sh
```

报告输出：

```text
projects/training/time-modeling/reports/tp_single_training_smoke/report.json
projects/training/time-modeling/reports/tp_multi_training_smoke/report.json
```

最近一次 `hanwu-proj` 实跑结果：

| 场景 | T_real | T_sim | 误差 |
|---|---:|---:|---:|
| 单机单卡 | `111.367228 ms` | `105.804344 ms` | `4.995081%` |
| 单机两卡 TP | `319.946518 ms` | `314.767113 ms` | `1.618835%` |

两组均低于 `20%` 误差目标。本次 `T_sim` 来自本地迁入的 `mvp_train_unified_estimator`，不是 profile 表。
