# 仓库结构规范

这个仓库按业务对象组织：算子、训练、推理、验收指标。硬件平台差异只放在脚本名、配置名和平台说明里，业务代码尽量复用。

## 顶层目录

```text
hanwu-proj/
  README.md
  docs/
    repo_structure.md
    history/
  projects/
    operators/
      communication/
      memory/
      compute/
    training/
      time-modeling/
    inference/
      time-modeling/
    indicators/
      1.2-runtime-validation/
      1.3-execution-model/
    shared/
      tools/
  scripts/
```

## 目录含义

- `projects/operators/communication/`：通信算子响应时间建模与验证。
- `projects/operators/memory/`：存储类算子建模与验证。
- `projects/operators/compute/`：计算类算子建模与验证。
- `projects/training/time-modeling/`：训练时间建模、训练 smoke、训练 Dashboard。
- `projects/inference/time-modeling/`：推理时间建模、推理 smoke、推理 Dashboard。
- `projects/indicators/1.2-runtime-validation/`：指标 1.2，训练/推理任务运行验证。
- `projects/indicators/1.3-execution-model/`：指标 1.3，训练/推理处理模型输出验证。
- `projects/shared/tools/`：跨项目复用的小工具。
- `scripts/`：仓库级统一入口，只做跳转和参数封装，不放核心业务逻辑。
- `docs/history/`：历史汇总、阶段进展、演示文档。

## 小项目内部约定

每个小项目尽量保持下面的形状：

```text
project/
  README.md
  scripts/      # 可执行入口
  configs/      # 可复用配置
  docs/         # 项目内设计和验证说明
  src/ or tools/# 业务代码和辅助工具
  database/     # 小型 profile 表或样例数据
  ui/           # Dashboard 前端
```

运行结果、临时模型、日志、图表和 Dashboard 运行目录默认不提交；需要交付的少量样例结果应在 README 中说明用途。

## 命名规则

目录名使用英文小写和连字符，表达业务含义，不使用个人名、日期串或拼音缩写作为主目录名。阶段日期可以保留在历史文档或报告文件名中。

新增硬件平台时按平台名出现在入口脚本或配置文件中：

- `run_*_cambricon_*.sh`
- `run_*_moorethreads_*.sh`
- `*_cambricon_*.json`
- `*_moorethreads_*.json`

同一类测试不要复制整套项目目录。优先新增平台配置和 wrapper 脚本，只有平台 SDK 强耦合的代码才单独拆文件。

## 仓库级入口

常用命令统一从根目录 `scripts/` 进入，例如：

```bash
bash scripts/run_inference_cambricon_dashboard.sh
bash scripts/run_training_cambricon_smoke.sh
bash scripts/run_operator_comm_cambricon.sh
```

小项目内原脚本继续保留，方便兼容已有文档和自动化。

## 清理规则

以下内容属于本地运行产物，不进入 Git：

- `tmp/`、`tmp_*/`
- `artifacts/`
- `reports/`
- `validation_reports/`
- `dashboard_runs/`
- `charts/`
- `build/`
- `__pycache__/`
- `*.log`
- LaTeX 中间文件
- 训练 checkpoint：`*.pt`、`*.pth`
