# 当前功能实现方案说明

本文档用于说明当前版本推理延迟估测工具的功能范围、实现结构与端到端执行流程，便于在汇报、评审和后续扩展时统一认识。

## 1. 项目定位

当前系统是一个基于 PyTorch 的推理性能分析 MVP，面向大语言模型推理场景，核心目标是回答三个问题：

- 给定模型、输入长度和并行配置后，`prefill`、`decode_step`、`request` 三个阶段预计耗时多少。
- 实际运行时，各阶段真实耗时多少，预测误差有多大。
- 时间主要消耗在哪些算子、模块和通信项上。

当前实现重点覆盖以下场景：

- 单卡推理。
- 单机多卡 Tensor Parallel 推理。
- 跨主机 Tensor Parallel 推理。
- 预测值与实测值的自动对比和报告输出。

## 2. 总体实现思路

系统采用“分析预测 + 在线实测 + 结果对齐”的闭环方案。

整体流程如下：

1. 解析执行参数，构建运行环境与并行配置。
2. 加载模型与输入，抽取 prefill/decode 两个推理阶段的图信息。
3. 基于硬件标定结果，对图中节点做解析式耗时估测。
4. 在 TP 场景下对模型做张量并行切分，并对估测结果做 shard-aware 缩放。
5. 结合模块级 profile 和通信预测，得到阶段级与请求级预测值。
6. 在线执行真实推理，采集阶段耗时、模块耗时和算子级 profile。
7. 输出 `report.json`、`report.md` 和 Dashboard 状态文件，形成可读结果。

对应主控入口为 `mvp_app.py`，命令入口为 `torch_infer_mvp.py`。

## 3. 功能模块与实现方案

### 3.1 执行配置与分布式初始化

该部分由 `mvp_execution.py` 负责，主要功能包括：

- 解析命令行参数，如模型路径、dtype、prompt、`tp_size`、`world_size`、输出目录等。
- 识别单卡或 TP 模式，并在 TP 模式下初始化 `torch.distributed`。
- 收集 rank、local rank、node rank、物理 GPU 映射和网络互联信息。
- 构造统一的 `ExecutionConfig`，供后续估测、实测和报告模块共享。

该设计的作用是把运行方式从业务逻辑中分离出来，保证单卡、单机多卡和跨主机多卡使用同一套主流程。

### 3.2 输入准备与推理图导出

该部分由 `mvp_runtime.py` 负责，主要实现包括：

- 使用 tokenizer 将文本 prompt 转成 `input_ids` 与 `attention_mask`。
- 先运行一次带 cache 的前向，构造 decode 阶段所需的 `next_token`、扩展后的 attention mask 和 `past_key_values`。
- 通过 `PrefillWrapper` 和 `DecodeWrapper` 将推理过程拆成两个可独立导出的阶段。
- 使用 `torch.export.export()` 分别导出 prefill 图和 decode 图。

当前系统没有直接把整次请求视为一个黑盒，而是显式拆成 `prefill` 和 `decode_step`，这样可以分别分析首 token 成本、单步生成成本以及总请求成本。

### 3.3 图缓存机制

为避免重复导图带来的额外开销，系统实现了图缓存机制，仍由 `mvp_runtime.py` 管理。

缓存键由以下因素共同决定：

- 模型标识。
- `dtype`。
- batch size。
- prompt token 数。
- 并行模式、`tp_size`、`world_size`、`nnodes`、互联类型。

命中缓存时，系统可直接复用之前保存的节点估测结果和图统计信息；未命中时重新导出并回写缓存。该机制用于降低重复实验时的准备成本，提升验证效率。

### 3.4 硬件标定与解析式估测

该部分由 `mvp_calibration.py` 和 `mvp_estimator.py` 协同完成。

实现方式为：

- 先执行 microbenchmark，得到当前设备上的 GEMM、attention、带宽、launch overhead 等标定参数。
- 遍历导出图中的节点，按算子类型、shape 和 phase 估算其 FLOPs、访存字节数和预计时间。
- 为每个节点生成统一的 `NodeEstimate`，作为后续聚合和对比的基础数据。

这一层提供的是“解析式基线估测”，优点是覆盖范围完整、无需对每个模型逐层实测即可先得到可解释的预测结果。

### 3.5 Tensor Parallel 切分与分布式感知缩放

该部分由 `mvp_parallel.py` 与 `mvp_graph.py` 完成。

当前 TP 实现面向 Llama 风格 decoder-only 模型，策略为：

- 对 `q_proj`、`k_proj`、`v_proj`、`gate_proj`、`up_proj` 做列切分。
- 对 `o_proj`、`down_proj` 做行切分。
- 同步调整 `num_heads` 与 `num_key_value_heads` 等结构参数。

由于推理图是在 TP 切分之前导出的，图中的节点默认仍反映单卡形态，因此系统增加了 `tp_shard_node_estimate()`，对相关估测结果按 TP 配置做 shard-aware 缩放，避免直接复用单卡估测导致系统性偏差。

### 3.6 模块级在线 Profile 与表驱动复用

该部分由 `mvp_profile.py` 与 `mvp_table.py` 提供支持，用来弥补纯解析式估测在复杂模块上的偏差。

当前支持三种模式：

- `online`：运行时直接采集模块 profile。
- `table`：仅从离线表数据库读取已有 profile。
- `hybrid`：优先查表，缺失部分再在线补采。

模块 profile 的核心做法是：

- 对 attention、MLP、decoder layer、norm、lm head 等作用域注册 hook。
- 通过 CUDA event 或 wall time 采集模块级耗时样本。
- 将模块 profile 与其覆盖的节点集合关联。
- 在阶段汇总时，用模块实测值替代对应节点估测值。

这一机制的意义是保留解析式方法的全局覆盖，同时引入模块级校正，提高阶段级预测精度。

### 3.7 阶段修正与通信预测

为进一步改善阶段级总时间，系统还实现了两类补偿机制：

- `collect_phase_adjustments()`：比较阶段总实测值与基础估测值，生成阶段修正量。
- `build_predicted_comm()`：根据 TP 配置、通信原语和消息规模启发式估计 collective 时间。

其中通信预测主要用于解释 TP 场景下的额外开销，阶段修正则用于降低解析式模型遗漏的运行时偏差。

### 3.8 真实运行测量

该部分由 `mvp_measurement.py` 负责，重点功能包括：

- 对 `prefill`、`decode_step`、`request` 进行真实执行计时。
- 支持 warmup 和 repeat 采样，输出 mean、median、min、max、samples。
- 在 TP 场景下通过 `dist.barrier()` 对齐各 rank 起止点。
- 以所有 rank 中的最大耗时作为该轮 phase latency。
- 分别保留聚合结果和各 rank 独立测量结果。

这一设计符合分布式推理的真实语义，因为端到端阶段耗时取决于最慢 rank，而不是单个 rank 的局部时间。

### 3.9 算子级对齐与误差分析

系统通过 `profile_cuda_ops()`、`build_operator_compare_rows()` 和 `compare_summary()` 实现算子级对齐分析。

具体做法是：

- 使用 `torch.profiler` 和 Dispatch 机制抓取 CUDA 算子序列与 collective 行为。
- 将测得的算子记录与估测侧节点按 target、scope、shape、ordinal 等字段进行匹配。
- 计算估测覆盖率、匹配率和各阶段相对误差。

这样可以从“总时间对不对”进一步下钻到“哪些算子估得准、哪些模块或通信项贡献了误差”。

### 3.10 报告与可视化状态输出

该部分由 `mvp_measurement.py` 和 `mvp_dashboard.py` 配合实现。

当前输出产物包括：

- `report.json`：完整结构化结果，便于程序消费。
- `report.md`：面向人工阅读的摘要报告。
- `dashboard_status.json`：供 Dashboard 或状态页面读取。

报告中包含的关键信息包括：

- 执行配置与拓扑信息。
- 图缓存命中情况。
- 各阶段预测值、实测值与相对误差。
- 模块级 profile、算子级对齐结果、通信汇总。
- 各 rank 的独立测量结果。

## 4. 当前版本的端到端流程

从代码主流程看，当前系统执行顺序可以概括为：

1. 参数解析与执行配置构建。
2. 硬件标定。
3. Tokenizer 编码与输入准备。
4. 读取图缓存；若未命中则加载模型、准备 runtime inputs 并导出 prefill/decode 图。
5. 对图节点做解析式估测。
6. 根据运行模式决定是否加载或采集模块 profile 与阶段修正。
7. 若为 TP 模式，对模型做张量并行切分并建立通信预测。
8. 真实执行 prefill、decode_step 和 request，得到阶段测量值。
9. 采集算子级 profile，完成估测与实测对齐。
10. 汇总误差、生成报告并写出 Dashboard 状态。

该流程同时支持“只估测不实跑”和“估测 + 实测验证”两种使用方式。

## 5. 当前已实现功能总结

从功能层面看，当前版本已经具备以下能力：

- 支持 LLM 推理阶段拆分为 `prefill` 与 `decode_step`。
- 支持单卡、单机 TP、跨主机 TP 三类执行场景。
- 支持基于 `torch.export` 的推理图提取。
- 支持节点级解析式性能估测。
- 支持 Llama 风格模型的 Tensor Parallel 切分。
- 支持模块级在线 profile、表查找和混合式 profile 复用。
- 支持分布式阶段计时与按最慢 rank 聚合。
- 支持算子级 predicted vs measured 对齐分析。
- 支持通信项识别、统计和预测。
- 支持图缓存、报告输出和 Dashboard 状态落盘。

## 6. 当前边界与已知限制

当前版本仍属于 MVP，边界比较明确：

- 主要面向 decoder-only LLM 推理，不覆盖完整训练流程。
- TP 切分目前按 Llama 风格结构实现，模型结构泛化能力有限。
- 图导出与估测逻辑围绕 PyTorch eager/`torch.export` 路径构建，未覆盖 `torch.compile` 或 CUDA Graph。
- 通信时间目前以启发式模型和实测汇总为主，还不是严格的高精度通信数据库方案。
- 模块级替换和阶段修正已具备实用价值，但仍依赖具体 scope 设计，跨模型迁移需要继续抽象。

## 7. 结论

当前系统已经形成一套可运行、可验证、可输出报告的推理延迟分析方案。其核心特点不是只做纯预测，也不是只做纯 profiling，而是把图分析、解析式估测、模块实测、通信估测和真实运行验证组合在一起，形成面向单卡与 TP 推理场景的闭环实现。

从工程角度看，当前版本已经具备继续扩展到更多模型、更多并行方式和更规范数据库体系的基础。
