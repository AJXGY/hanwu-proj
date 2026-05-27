# 托管环境执行手册

## 目标

避免重复先前出现的故障模式：在无中间输出的前台启动长时间运行任务，迫使我们必须在不确定任务是否实际卡死、仍在估算或已完成并正在生成大型报告的情况下，手动中断任务。

## 规则

1. 不要以前台内联 Python heredoc 方式运行长时间估算/测量任务并静默等待。
2. 在后台启动长时间任务。
3. 将 stdout/stderr 重定向到日志文件。
4. 启动后立即写入小型元数据文件。
5. 从输出目录轮询 `dashboard_status.json` 和 `report.json`，而非等待父辅助进程。
6. 将 `dashboard_status.json` 视作主要进度信号。
7. 若输出产物已包含最终计时数据，直接读取并停止等待辅助进程。
8. 若进度在异常时长内无进展，停止并检查产物后再启动新运行。

## 标准工作流

### 1. 预检查

在任何重新运行前执行以下检查：

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader
python3 tools/dashboard_env.py status --config configs/dashboard_env.json
pgrep -af "nvidia-cuda-mps-server|nvidia-cuda-mps-control"
```

解读说明：

- 若 GPU 计算应用非空，在明确其用途前不要启动新运行。
- 若仅残留陈旧的 `nvidia-cuda-mps-server`，先停止托管环境并清理。
- 若不再需要托管环境，在清理前先停止它。

### 2. 清理

推荐清理顺序：

```bash
python3 tools/dashboard_env.py stop --config configs/dashboard_env.json
pkill -f nvidia-cuda-mps-server
pkill -f nvidia-cuda-mps-control
```

注意事项：

- `nvidia-cuda-mps-server` 是释放 GPU 计算状态的关键进程。
- 部分 `nvidia-cuda-mps-control -d` 守护进程可能属于其他用户；若 `kill` 返回 `Operation not permitted`，记录该情况并在 GPU 计算应用已清空时继续。
- 清理后使用 `nvidia-smi --query-compute-apps=...` 重新检查。

### 3. 一次性重新准备

```bash
python3 tools/dashboard_env.py prepare --config configs/dashboard_env.json
docker inspect -f '{{.Id}} {{.State.Running}}' mvp-dashboard-env
```

记录容器 ID。同一批次的所有重新运行必须保持相同容器 ID。若 ID 意外变化，应停止并调查，避免混合来自不同准备环境的结果。

### 4. 在后台启动长时间运行

必需行为：

- 生成唯一运行 ID。
- 立即在 `/tmp` 下创建元文件。
- 将 stdout/stderr 重定向至 `/tmp/<run_id>.log`。
- 保留摘要文件路径，但不将其作为唯一完成信号等待。

需立即持久化的最小元数据：

- `run_id`
- `created_at`
- `output_dir`
- `summary_path`
- `container_before`

### 5. 轮询而非静默等待

轮询顺序：

1. 读取元文件获取 `output_dir`。
2. 轮询 `output_dir/dashboard_status.json`。
3. 在 `estimation_wall_time_s`、`measurement_wall_time_s`、`predictor_total_wall_time_s` 出现后立即读取。
4. 从 `report.json` 读取 `request_end_to_end_time_ms`、实测请求均值及相对误差。
5. 仅将父进程状态作为次要信号。

近期运行的关键观察：

- 辅助父进程可能在 `dashboard_status.json` 已达 `measurement_ready` 且所有有效计时数据已存入磁盘后仍保持活动状态。
- 因此，产物轮询比等待辅助进程退出更可靠。

### 6. 何时停止并分析

当发生以下任一情况时，立即停止并检查产物/日志：

- 准备的容器 ID 意外变化。
- 在明显完成阶段后仍缺失 `report.json`。
- `stderr` 显示 CUDA 设备不匹配、MPS 启动冲突或分布式初始化失败。
- 运行在同一阶段停留过久且无文件增长。
- 准备启动新运行时，`nvidia-smi` 中仍可见旧计算应用。

## 结果收集

每次运行至少记录：

- 运行前后的容器 ID
- 输出目录
- 估算墙钟时间
- 测量墙钟时间
- 预测器总墙钟时间
- 预估请求延迟
- 实测请求延迟
- 预填充相对误差
- 解码步相对误差
- 请求相对误差

## 推荐模式（适用于 1/2/4 批次运行）

1. 预检查 GPU 和托管环境。
2. 清理陈旧 MPS 状态。
3. 一次性准备环境。
4. 记录容器 ID。
5. 每次在后台启动一个运行。
6. 轮询产物直至获得计时数据和对比结果。
7. 若产物已完成但辅助进程仍存活，终止残留的辅助父进程。
8. 确认容器 ID 未变化。
9. 转入下一 GPU 数量配置。

## 当前已知问题

- 复用容器可能继承陈旧 MPS 状态，除非任务路径在每次运行前显式重启 MPS。
- 前台 Python heredoc 运行不适合长时间报告生成，因其不提供进度可见性。
- 大型 `report.json` 文件可能已完成，而包装进程仍在最终处理。
- 当代码工作树存在未提交更改时，输出当前会写入 `/tmp/0324proj-output/...`。