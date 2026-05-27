# 当前训练逻辑对比：hanwu-proj vs train-infer-estimation

本文对比当前代码状态：

- 当前训练代码：`/home/o_mabin/hanwu-proj/projects/training/time-modeling`
- 参考训练代码：`/home/o_mabin/train-infer-estimation`

结论：除平台适配和跨机真实训练执行外，`hanwu-proj` 的训练预测框架已经按 `train-infer-estimation` 的方式迁移。核心训练估算不再是简化仿写，而是本地同名 `mvp_*` 模块链路：`mvp_train_unified_estimator -> mvp_train_tp_estimator / mvp_backward_comm / mvp_optimizer_tp_estimator / mvp_train_estimator`。

## 1. 当前入口

主训练入口：

```text
projects/training/time-modeling/torch_train_tp_mvp.py
```

本地迁入的 `train-infer-estimation` 核心模块：

```text
mvp_train_unified_estimator.py
mvp_train_tp_estimator.py
mvp_backward_comm.py
mvp_optimizer_tp_estimator.py
mvp_train_estimator.py
mvp_train_types.py
mvp_train_graph.py
mvp_backward_graph.py
mvp_graph.py
mvp_types.py
lora_adapter.py
```

寒武纪平台适配模块：

```text
tp_backend.py
train_infer_static_estimator.py
configs/train_config.yaml
```

脚本入口：

```text
scripts/run_training_single_cambricon.sh
scripts/run_training_multi_cambricon.sh
scripts/run_training_cambricon_smoke.sh
```

## 2. 对齐情况

| 维度 | hanwu-proj 当前训练 | train-infer-estimation |
|---|---|---|
| 估算来源 | 静态图/FLOPs/bytes + 校准 | 静态图/FLOPs/bytes + 校准 |
| 核心 estimator | `mvp_train_unified_estimator` | `mvp_train_unified_estimator` |
| Forward TP | `mvp_train_tp_estimator` | `mvp_train_tp_estimator` |
| Backward comm | `mvp_backward_comm` | `mvp_backward_comm` |
| Optimizer TP | `mvp_optimizer_tp_estimator` | `mvp_optimizer_tp_estimator` |
| 分解粒度 | `forward / backward_compute / backward_comm / optimizer` | `forward / backward_compute / backward_comm / optimizer` |
| 校准方式 | `configs/train_config.yaml` 四阶段校准参数 | `config/train_config.yaml` 四阶段校准参数 |
| DDP 估算模块 | 模块已迁入；当前主线不跑 DDP | 原生支持 |
| 图抽取模块 | 已迁入同名模块 | 原生支持 |
| 训练负载 | `lora_vocab_adapter` | `lora_vocab_adapter` |
| Backbone backward | 不冻结，autograd 穿过 backbone | 不冻结，autograd 穿过 backbone |
| Adapter 结构 | `hidden_size -> rank -> vocab_size` | `hidden_size -> rank -> vocab_size` |
| Loss | token-level vocab CE | token-level vocab CE |
| Optimizer 参数 | 只更新 adapter 参数 | 只更新 adapter 参数 |
| 单机单卡 | 已真实跑通 | 支持 |
| 单机多卡 TP | 已真实跑通 | 支持 |
| 跨机真实训练执行 | 暂不作为主线验证 | 有参数结构 |
| 平台后端 | 寒武纪 MLU/CNCL | NVIDIA CUDA/NCCL |

## 3. 仍保留的差异

当前只保留两类差异：

- 平台差异：`hanwu-proj` 的真实运行使用 MLU/CNCL、寒武纪 Docker 和 MLU 同步/缓存接口；`train-infer-estimation` 原项目默认 CUDA/NCCL。
- 跨机执行差异：跨机参数和估算模块已保留，但本轮真实 smoke 只验证单机单卡和单机两卡 TP。

已去掉旧寒武纪 classifier 路径的关键差异：

- 不再冻结 backbone。
- 不再用 `no_grad` 包住 backbone forward。
- 不再对 hidden state 做 `.detach()` 后训练小 head。
- 不再使用 `hidden_size -> rank -> num_labels` 的 2 类分类 head。
- 不再使用 pooled hidden state 分类 loss。

当前 `Llama-3.1-8B`、`lora_rank=8` 的 adapter 参数量是：

```text
8 * (4096 + 128256) = 1,058,816
```

## 4. 当前真实运行结果

最近一次运行命令：

```bash
SEQUENCE_LENGTH=8 MICROBATCH_COUNT=1 bash scripts/run_training_cambricon_smoke.sh
```

结果：

| 场景 | 模式 | estimator | T_real | T_sim | 误差 |
|---|---|---|---:|---:|---:|
| 单机单卡 | `single`, `tp_size=1` | `mvp_train_unified_estimator` | `111.367228 ms` | `105.804344 ms` | `4.995081%` |
| 单机两卡 | `tp`, `tp_size=2` | `mvp_train_unified_estimator` | `319.946518 ms` | `314.767113 ms` | `1.618835%` |

阶段估算：

| 场景 | forward | backward_compute | backward_comm | optimizer |
|---|---:|---:|---:|---:|
| 单机单卡 | `30.278853 ms` | `75.433257 ms` | `0.000000 ms` | `0.092235 ms` |
| 单机两卡 TP | `120.183168 ms` | `180.522066 ms` | `13.999469 ms` | `0.062411 ms` |

报告文件：

```text
reports/tp_single_training_smoke/report.json
reports/tp_multi_training_smoke/report.json
```

两组误差都低于 `20%`。

## 5. 当前结论

现在 `hanwu-proj` 和 `train-infer-estimation` 的训练预测方法已经对齐到同一类训练任务和同一套估算链路。`hanwu-proj` 只在寒武纪平台执行、设备后端、脚本包装和跨机真实运行验证范围上不同；训练估算核心已迁为同名模块和同类报告字段。
