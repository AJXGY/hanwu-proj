# 项目目录说明

## 文档

- `0409任务.md`：本次通信密集型算子建模测试任务说明
- `0409进展.md`：当前进展、阶段性结论与阻塞说明
- `all_task.md`：整体测试项说明
- `寒武纪GPU docker环境使用指南.md`：Docker 运行方法

## 代码

- `scripts/cambricon_comm_bench.py`
  - PyTorch CNCL 通信微基准原型
- `scripts/cncl_comm_bench.cc`
  - CNRT/CNCL 原生通信微基准原型
- `scripts/parse_cncl_benchmark.py`
  - 官方 benchmark 日志转 CSV
- `scripts/analyze_comm_results.py`
  - 建模拟合、误差统计与图片生成
- `scripts/run_official_cncl_bench_docker.sh`
  - 一键执行 Docker benchmark、日志解析和图表生成
- `scripts/comm_response_time_tool.py`
  - 自研任务响应时间分析工具原型，支持建模、单点预测和严格版验证
- `scripts/run_comm_response_time_tool_validation.sh`
  - 一键执行严格版 D-F 流程

## 结果

- `results/raw/`
  - 原始 benchmark 日志
- `results/processed/`
  - 解析后的 CSV、严格版模型文件与误差统计表
- `figure/`
  - 图表输出
  - `figure/strict/` 为严格版验证图片

## 编译产物

- `build/`
  - 生成的可执行文件

## 关键运行说明

- 正式双卡测试时，Docker 必须增加：
  - `--ipc=host`
  - `--cgroupns=host`
- 否则容器里虽然可能看到 2 张卡，但 `CNRT/CNCL` 会有效退化成单卡。
