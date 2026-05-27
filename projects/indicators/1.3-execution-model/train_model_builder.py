from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Cambricon training execution model artifacts"
    )
    parser.add_argument("--train-repo", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--pp-size", type=int, choices=[1, 2], default=1)
    parser.add_argument("--microbatch-count", type=int, default=2)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--physical-devices", default="0")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--adapter-num-labels", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def render_svg(nodes: list[dict], edges: list[tuple[str, str]], title: str) -> str:
    width = 1640
    height = 500
    positions = {}
    for idx, node in enumerate(nodes):
        positions[node["id"]] = (130 + idx * 190, 120 + node.get("lane", 0) * 110)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#64748b"/></marker></defs>',
        f'<text x="40" y="42" font-size="28" font-family="sans-serif" fill="#111827">{title}</text>',
    ]
    for src, dst in edges:
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        lines.append(
            f'<line x1="{x1 + 70}" y1="{y1}" x2="{x2 - 70}" y2="{y2}" stroke="#64748b" stroke-width="3" marker-end="url(#arrow)"/>'
        )
    for node in nodes:
        x, y = positions[node["id"]]
        kind = node.get("kind", "compute")
        fill = "#dbeafe" if kind == "compute" else "#fee2e2" if kind == "comm" else "#ecfccb"
        lines.extend(
            [
                f'<rect x="{x - 70}" y="{y - 34}" width="140" height="68" rx="18" fill="{fill}" stroke="#1f2937" stroke-width="2"/>',
                f'<text x="{x}" y="{y - 6}" text-anchor="middle" font-size="16" font-family="sans-serif" fill="#111827">{node["label"]}</text>',
                f'<text x="{x}" y="{y + 16}" text-anchor="middle" font-size="11" font-family="sans-serif" fill="#475569">{node.get("sub", "")}</text>',
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    train_repo = Path(args.train_repo).expanduser().resolve()
    sys.path.insert(0, str(train_repo / "src"))
    from transformers import AutoConfig
    from train0411_clj.train_pipeline_mvp import stage_layer_range

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = AutoConfig.from_pretrained(args.model_path)
    total_layers = int(getattr(model_config, "num_hidden_layers", 0))
    physical_devices = [int(item.strip()) for item in args.physical_devices.split(",") if item.strip()]

    stage_ranges = []
    for stage_index in range(args.pp_size):
        start, end = stage_layer_range(total_layers, stage_index, args.pp_size)
        stage_ranges.append(
            {
                "stage_index": stage_index,
                "physical_device": physical_devices[stage_index],
                "layer_range": [start, end - 1],
                "layer_count": end - start,
            }
        )

    nodes = [{"id": "cpu_load", "label": "CPU 数据准备", "sub": "batch assemble", "lane": 0, "kind": "cpu"}]
    edges: list[tuple[str, str]] = []
    previous_tail = "cpu_load"
    for microbatch_idx in range(args.microbatch_count):
        if args.pp_size == 1:
            node_id = f"mb{microbatch_idx}_stage0"
            nodes.append(
                {
                    "id": node_id,
                    "label": f"MB{microbatch_idx} Backbone Forward",
                    "sub": f"frozen layers 0-{total_layers - 1}",
                    "lane": 1,
                    "kind": "compute",
                }
            )
            edges.append((previous_tail, node_id))
            previous_tail = node_id
        else:
            s0_id = f"mb{microbatch_idx}_stage0"
            s1_id = f"mb{microbatch_idx}_stage1"
            comm_id = f"mb{microbatch_idx}_comm"
            nodes.extend(
                [
                    {
                        "id": s0_id,
                        "label": f"MB{microbatch_idx} Stage0 Forward",
                        "sub": f"frozen layers {stage_ranges[0]['layer_range'][0]}-{stage_ranges[0]['layer_range'][1]}",
                        "lane": 1,
                        "kind": "compute",
                    },
                    {
                        "id": comm_id,
                        "label": "Stage 传输",
                        "sub": "hidden_states",
                        "lane": 2,
                        "kind": "comm",
                    },
                    {
                        "id": s1_id,
                        "label": f"MB{microbatch_idx} Stage1 + Adapter",
                        "sub": f"frozen layers {stage_ranges[1]['layer_range'][0]}-{stage_ranges[1]['layer_range'][1]} + low-rank head",
                        "lane": 3,
                        "kind": "compute",
                    },
                ]
            )
            edges.extend([(previous_tail, s0_id), (s0_id, comm_id), (comm_id, s1_id)])
            previous_tail = s1_id
    nodes.append({"id": "optimizer", "label": "Optimizer Step", "sub": "adapter parameters only", "lane": 2, "kind": "compute"})
    edges.append((previous_tail, "optimizer"))
    nodes.append({"id": "checkpoint", "label": "Checkpoint", "sub": "save artifacts", "lane": 0, "kind": "cpu"})
    edges.append(("optimizer", "checkpoint"))

    resource_mapping = {
        "cpu_roles": [
            "dataset / batch assembly",
            "launch orchestration",
            "checkpoint metadata save",
        ],
        "npu_roles": [
            {
                "stage_index": item["stage_index"],
                "physical_device": item["physical_device"],
                "assignment": (
                    "frozen backbone forward + adapter update"
                    if args.pp_size == 1
                    else f"pipeline stage {item['stage_index']} frozen backbone"
                ),
            }
            for item in stage_ranges
        ],
    }
    execution_logic = {
        "parallel_mode": "single_stage" if args.pp_size == 1 else "pipeline_parallel",
        "microbatch_count": args.microbatch_count,
        "microbatch_logic": (
            f"{args.microbatch_count} microbatches execute serially on one stage"
            if args.pp_size == 1
            else f"{args.microbatch_count} microbatches flow across {args.pp_size} pipeline stages"
        ),
        "training_mode": "lora_style_adapter",
        "backbone_frozen": True,
        "adapter": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "num_labels": args.adapter_num_labels,
        },
        "optimizer_step": "all microbatches finish adapter backward before optimizer step",
    }
    execution_model = {
        "task": "training_model_structure",
        "model_path": args.model_path,
        "model_type": getattr(model_config, "model_type", "unknown"),
        "num_hidden_layers": total_layers,
        "hidden_size": int(getattr(model_config, "hidden_size", 0)),
        "intermediate_size": int(getattr(model_config, "intermediate_size", 0)),
        "num_attention_heads": int(getattr(model_config, "num_attention_heads", 0)),
        "vocab_size": int(getattr(model_config, "vocab_size", 0)),
        "training_mode": "lora_style_adapter",
        "backbone_frozen": True,
        "adapter": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "num_labels": args.adapter_num_labels,
        },
        "resource_mapping": resource_mapping,
        "stage_partitioning": stage_ranges,
        "execution_logic": execution_logic,
        "dag": {"nodes": nodes, "edges": edges},
        "artifacts": {
            "execution_dag_svg": str(output_dir / "execution_dag.svg"),
            "execution_dag_json": str(output_dir / "execution_dag.json"),
        },
    }
    (output_dir / "execution_model.json").write_text(
        json.dumps(execution_model, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "execution_dag.json").write_text(
        json.dumps({"nodes": nodes, "edges": edges}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "execution_dag.svg").write_text(
        render_svg(nodes, edges, "Training Execution DAG"),
        encoding="utf-8",
    )
    index_html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/><title>Training Model Structure</title></head>
<body>
<h1>Training Model Structure</h1>
<p><a href="execution_model.json">execution_model.json</a></p>
<p><a href="execution_dag.svg">execution_dag.svg</a></p>
<img src="execution_dag.svg" style="max-width:100%;border:1px solid #ddd"/>
</body></html>"""
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")
    summary = {
        "task": "training_model_structure",
        "success": True,
        "pp_size": args.pp_size,
        "physical_devices": physical_devices,
        "microbatch_count": args.microbatch_count,
        "execution_model_path": str(output_dir / "execution_model.json"),
        "dag_svg_path": str(output_dir / "execution_dag.svg"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
