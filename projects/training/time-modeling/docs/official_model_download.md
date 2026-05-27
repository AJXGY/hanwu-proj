# 官方模型下载说明

`Meta-Llama-3.1-8B` 官方 Hugging Face 仓库是 gated repo。

我在这台机器上实际验证过，无凭证访问会返回：

- HTTP `401`
- `x-error-code: GatedRepo`

因此没有已经授权的 Hugging Face 账号/token 时，不能直接把官方权重下载到本机。

## 下载前提

1. 你的 Hugging Face 账号已经在官方模型页接受 Meta Llama 3.1 license
2. 本机有对应的 `HF_TOKEN`

官方模型页：

- https://huggingface.co/meta-llama/Meta-Llama-3.1-8B

## 下载命令

在工作区根目录执行：

```bash
HF_TOKEN=你的token ./scripts/download_official_llama31_8b.sh
```

默认下载目录：

```text
models/Llama-3.1-8B
```

下载完成后，训练命令里的 `--model-path` 可直接改成：

```bash
--model-path /workspace/models/Llama-3.1-8B
```
