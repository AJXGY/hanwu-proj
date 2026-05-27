# Hanwuji Training TP LoRA-style Report

## Conclusion

Hanwuji training should use the TP training path as the default path going forward:

- Runtime entry: `torch_train_tp_mvp.py`
- Parallel mode: `--parallel-mode tp`
- Launch method: `python -m torch.distributed.run`
- Tuning mode: `lora_style_adapter`
- Backbone policy: frozen Llama backbone, adapter/head parameters only

The previous PP training smoke path remains useful as a compatibility or comparison path, but the training validation baseline should be TP.

## Current TP Smoke Result

Run date: 2026-05-07

Model: `smoke_models/tiny_llama_mlu`

Command shape:

```bash
python -m torch.distributed.run \
  --nproc_per_node 2 \
  --master_addr 127.0.0.1 \
  --master_port 29531 \
  torch_train_tp_mvp.py \
  --model-path smoke_models/tiny_llama_mlu \
  --device mlu:0 \
  --parallel-mode tp \
  --physical-devices 0,1 \
  --world-size 2 \
  --tp-size 2 \
  --nproc-per-node 2 \
  --microbatch-count 2 \
  --microbatch-size 1 \
  --sequence-length 8 \
  --warmup 0 \
  --benchmark-repeat 1 \
  --profile-repeat 1 \
  --optimizer-type sgd \
  --output-dir /workspace/tmp_lora_style_tp2_mlu
```

Key result:

| Field | Value |
| --- | --- |
| accelerator_kind | `mlu` |
| parallel_mode | `tp` |
| world_size | `2` |
| tp_size | `2` |
| nproc_per_node | `2` |
| training_mode | `lora_style_adapter` |
| backbone_frozen | `true` |
| lora_rank | `8` |
| lora_alpha | `16.0` |
| adapter_num_labels | `2` |
| measured_train_iteration_ms | `31.955976` |
| estimated_train_iteration_ms | `34.678074` |
| relative_error_pct | `8.518274` |

Output artifacts:

- `hanwu-proj/tmp_lora_style_tp2_mlu/report.json`
- `hanwu-proj/tmp_lora_style_tp2_mlu/report.md`

## Rank Placement

| Rank | Local Rank | Physical Device |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 1 | 1 |

Per-rank train iteration timing:

| Rank | Device | mean_ms |
| --- | --- | --- |
| 0 | 0 | `31.555485` |
| 1 | 1 | `31.955976` |

## Implementation State

The TP training implementation uses the same LoRA-style policy as the rest of the updated training code:

- Load Llama with `AutoModelForCausalLM`
- Apply TP sharding to Llama layers when `--parallel-mode tp`
- Freeze the Llama backbone
- Train only a low-rank adapter/classification head
- Report `training_mode = lora_style_adapter`
- Report `backbone_frozen = true`

Related files:

- `projects/inference/time-modeling/torch_train_tp_mvp.py`
- `projects/training/time-modeling/src/train0411_clj/train_pipeline_mvp.py`
- `projects/indicators/1.2-runtime-validation/train_task_runner.py`
- `projects/indicators/1.3-execution-model/train_model_builder.py`

## Recommendation

Use TP as the default training validation route:

```text
torch_train_tp_mvp.py --parallel-mode tp --world-size N --tp-size N --nproc-per-node N
```

For local two-card validation, use `N=2` and `--physical-devices 0,1`.
