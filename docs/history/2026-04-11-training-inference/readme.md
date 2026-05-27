**1) 只拿源码时，先准备什么**
- 本机准备一个可用 Python（建议你现在用的）：`/home/o_zhanghui/miniconda3/envs/llama_4gpu/bin/python`
- 这个环境里至少要能跑 `torch_infer_mvp.py`（PyTorch/Transformers 等）
- 模型目录要存在（本机）
- 如果要跨机：
  - 主机到远端要免密 SSH（因为代码里用 `ssh -o BatchMode=yes`）
  - 远端也要有代码和 Python；并且路径尽量和主机一致
  - 远端模型路径可单独填 `remote_model_path`

---

**2) 启动 Dashboard**
在仓库根目录执行：

```bash
python mvp_dashboard.py
```

打开：`http://127.0.0.1:8123`

可选环境变量：
- `MVP_DASHBOARD_CONFIG`：指定配置文件路径
- `MVP_DASHBOARD_HOST` / `MVP_DASHBOARD_PORT`：改监听地址端口

---

**3) 启动后怎么操作**

**A. 单机**
- 保持 `部署模式=单机`
- 选 `单机几卡`
- 检查模型路径、`warmup/benchmark_repeat/profile_repeat`
- 点 `启动分析`

---

可以让AI帮你们配置环境，有啥部署、运行问题再来问我？

单机除了webui有点卡顿应该可以运行，跨机配置很麻烦，不过寒武纪好像只用单机吧，你们先试试看

---

**4) 新增：寒武纪 PP + microbatch 推理脚本**

仓库里现在已经补了一条独立的 pipeline parallel 推理入口：

- 主脚本：`torch_infer_pipeline_mvp.py`
- 运行实现：`mvp_pipeline_infer.py`
- 寒武纪 smoke 脚本：`scripts/run_cambricon_pp_smoke.sh`

当前已支持：

- `pp_size=1/2`
- `microbatch_count=1/2/4`
- 单机双卡寒武纪 `MLU580`
- 默认真实模型：`/home/o_mabin/LLM/models/Llama-3.1-8B`

当前真实 8B 推理结果：

- `pp1_mb2`: `error=0.6428%`
- `pp2_mb1`: `error=0.0000%`
- `pp2_mb2`: `error=1.7909%`
- `pp2_mb4`: `error=2.9232%`

直接运行：

```bash
cd /home/o_mabin/hanwu-proj/projects/inference/time-modeling
bash scripts/run_cambricon_pp_smoke.sh
```

会产出：

- `validation_reports/cambricon_pp_smoke/pp1_mb2/report.json`
- `validation_reports/cambricon_pp_smoke/pp2_mb1/report.json`
- `validation_reports/cambricon_pp_smoke/pp2_mb2/report.json`
- `validation_reports/cambricon_pp_smoke/pp2_mb4/report.json`
