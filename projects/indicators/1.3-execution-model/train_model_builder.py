from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Cambricon TP training execution model artifacts"
    )
    parser.add_argument("--train-repo", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--parallel-mode", choices=["single", "tp"], default="single")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--microbatch-count", type=int, default=2)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--physical-devices", default="0")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--adapter-num-labels", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_physical_devices(raw_value: str) -> list[int]:
    devices = [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    return devices or [0]


def render_svg(nodes: list[dict], edges: list[tuple[str, str]], title: str) -> str:
    width = max(1640, 260 + len(nodes) * 170)
    height = 500
    positions = {}
    for idx, node in enumerate(nodes):
        positions[node["id"]] = (130 + idx * 170, 110 + node.get("lane", 0) * 105)
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
                f'<rect x="{x - 70}" y="{y - 34}" width="140" height="68" rx="12" fill="{fill}" stroke="#1f2937" stroke-width="2"/>',
                f'<text x="{x}" y="{y - 6}" text-anchor="middle" font-size="15" font-family="sans-serif" fill="#111827">{node["label"]}</text>',
                f'<text x="{x}" y="{y + 16}" text-anchor="middle" font-size="11" font-family="sans-serif" fill="#475569">{node.get("sub", "")}</text>',
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines)


def build_dag(args: argparse.Namespace, total_layers: int, parallel_mode: str) -> tuple[list[dict], list[tuple[str, str]]]:
    nodes = [{"id": "cpu_load", "label": "CPU Data", "sub": "batch assemble", "lane": 0, "kind": "cpu"}]
    edges: list[tuple[str, str]] = []
    previous_tail = "cpu_load"
    for microbatch_idx in range(args.microbatch_count):
        forward_id = f"mb{microbatch_idx}_forward"
        nodes.append(
            {
                "id": forward_id,
                "label": f"MB{microbatch_idx} Forward",
                "sub": f"{parallel_mode.upper()} layers 0-{total_layers - 1}",
                "lane": 1,
                "kind": "compute",
            }
        )
        edges.append((previous_tail, forward_id))
        previous_tail = forward_id
        if parallel_mode == "tp":
            forward_comm_id = f"mb{microbatch_idx}_forward_sync"
            nodes.append(
                {
                    "id": forward_comm_id,
                    "label": "TP Sync",
                    "sub": "collective activations",
                    "lane": 2,
                    "kind": "comm",
                }
            )
            edges.append((previous_tail, forward_comm_id))
            previous_tail = forward_comm_id
        backward_id = f"mb{microbatch_idx}_backward"
        nodes.append(
            {
                "id": backward_id,
                "label": f"MB{microbatch_idx} Backward",
                "sub": "adapter gradients",
                "lane": 1,
                "kind": "compute",
            }
        )
        edges.append((previous_tail, backward_id))
        previous_tail = backward_id
        if parallel_mode == "tp":
            backward_comm_id = f"mb{microbatch_idx}_grad_sync"
            nodes.append(
                {
                    "id": backward_comm_id,
                    "label": "Grad Sync",
                    "sub": "TP all-reduce",
                    "lane": 2,
                    "kind": "comm",
                }
            )
            edges.append((previous_tail, backward_comm_id))
            previous_tail = backward_comm_id
    nodes.append({"id": "optimizer", "label": "Optimizer", "sub": "adapter parameters", "lane": 1, "kind": "compute"})
    edges.append((previous_tail, "optimizer"))
    nodes.append({"id": "checkpoint", "label": "Checkpoint", "sub": "save artifacts", "lane": 0, "kind": "cpu"})
    edges.append(("optimizer", "checkpoint"))
    return nodes, edges


def main() -> None:
    args = parse_args()
    train_repo = Path(args.train_repo).expanduser().resolve()
    sys.path.insert(0, str(train_repo / "src"))
    from transformers import AutoConfig

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = AutoConfig.from_pretrained(args.model_path)
    total_layers = int(getattr(model_config, "num_hidden_layers", 0))
    physical_devices = parse_physical_devices(args.physical_devices)
    parallel_mode = "tp" if args.parallel_mode == "tp" else "single"
    world_size = max(1, int(args.world_size))
    tp_size = max(1, int(args.tp_size))
    if parallel_mode == "tp":
        world_size = max(world_size, tp_size, min(len(physical_devices), tp_size))
        if tp_size < 2:
            raise RuntimeError("parallel-mode=tp requires tp-size >= 2")
        if len(physical_devices) < tp_size:
            raise RuntimeError(
                f"tp_size={tp_size} requires at least {tp_size} physical devices"
            )
    else:
        world_size = 1
        tp_size = 1

    nodes, edges = build_dag(args, total_layers, parallel_mode)
    tensor_parallel_partitioning = [
        {
            "rank": rank,
            "physical_device": physical_devices[rank],
            "tp_size": tp_size,
            "layer_range": [0, total_layers - 1],
            "assignment": (
                "full training graph"
                if parallel_mode == "single"
                else f"tensor-parallel shard {rank}/{tp_size} for attention, MLP, and adapter update"
            ),
        }
        for rank in range(tp_size)
    ]
    resource_mapping = {
        "cpu_roles": [
            "dataset / batch assembly",
            "launch orchestration",
            "checkpoint metadata save",
        ],
        "npu_roles": tensor_parallel_partitioning,
    }
    execution_logic = {
        "parallel_mode": parallel_mode,
        "world_size": world_size,
        "tp_size": tp_size,
        "microbatch_count": args.microbatch_count,
        "microbatch_logic": (
            f"{args.microbatch_count} microbatches execute serially on one device"
            if parallel_mode == "single"
            else f"{args.microbatch_count} microbatches execute with tensor-parallel synchronization on each step"
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
        "tensor_parallel_partitioning": tensor_parallel_partitioning,
        "stage_partitioning": [],
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
        render_svg(nodes, edges, "Training TP Execution DAG"),
        encoding="utf-8",
    )
    index_html = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/><title>Training TP Model Structure</title></head>
<body>
<h1>Training TP Model Structure</h1>
<p><a href="execution_model.json">execution_model.json</a></p>
<p><a href="execution_dag.svg">execution_dag.svg</a></p>
<img src="execution_dag.svg" style="max-width:100%;border:1px solid #ddd"/>
</body></html>"""
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")
    summary = {
        "task": "training_model_structure",
        "success": True,
        "parallel_mode": parallel_mode,
        "world_size": world_size,
        "tp_size": tp_size,
        "physical_devices": physical_devices[:tp_size],
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
