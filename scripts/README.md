# 统一脚本入口

这里放仓库级 wrapper。脚本只负责进入对应项目目录并调用项目内脚本，核心逻辑仍放在各小项目中。

命名规则：

- `run_operator_*_<platform>.sh`
- `run_training_single_<platform>.sh`
- `run_training_multi_<platform>.sh`
- `run_inference_single_<platform>.sh`
- `run_inference_multi_<platform>.sh`
- `run_indicator_*_dashboard.sh`

训练和推理统一使用 TP 路线；PP 脚本只作为历史实验保留在项目内部，不作为标准入口。

当前平台名：

- `cambricon`：寒武纪
- `moorethreads`：摩尔线程，新增时沿用同一命名方式
