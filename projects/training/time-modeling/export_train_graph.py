#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for candidate in [
    HERE,
    HERE / "train_infer_estimation_core",
    HERE.parent / "training" / "time-modeling" / "train_runtime" / "train_infer_estimation_core",
    HERE.parent / "training" / "time-modeling" / "train_runtime",
    Path("/home/o_mabin/moer-proj/projects/training/time-modeling/train_runtime/train_infer_estimation_core"),
    Path("/home/o_mabin/moer-proj/projects/training/time-modeling/train_runtime"),
]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    import torch_mlu  # noqa: F401
except Exception:
    pass
try:
    import torch_musa  # noqa: F401
except Exception:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mvp_runtime import prepare_inputs
from mvp_backward_graph import extract_backward_graph, get_gradient_summary

COLORS = {
    "input": "#1d4ed8",
    "embedding": "#0f766e",
    "layer": "#475569",
    "norm": "#7c3aed",
    "lm_head": "#15803d",
    "gradient": "#dc2626",
    "output": "#111827",
    "misc": "#334155",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export training backward layer graph")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", default="alpha alpha alpha alpha alpha alpha alpha alpha")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=0)  # compatibility with NVIDIA CLI
    parser.add_argument("--repeat", type=int, default=1)  # compatibility with NVIDIA CLI
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backward-max-nodes", type=int, default=220)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def shorten(value: Any, limit: int = 34) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: max(limit - 3, 1)] + "..."


def layer_index(name: str) -> int | None:
    match = re.search(r"(?:^|\.)(?:model\.)?layers\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


def group_for_gradient(name: str) -> tuple[str, str, str, int]:
    normalized = name.replace("model.model.", "model.")
    idx = layer_index(normalized)
    if "embed_tokens" in normalized or "embedding" in normalized:
        return "embed_tokens", "embed", "embedding", 10_000
    if idx is not None:
        return f"model.layers.{idx}", f"layer {idx:02d}", "layer", 9_000 - idx
    if "lm_head" in normalized:
        return "lm_head", "lm_head", "lm_head", 0
    if normalized.endswith("norm") or ".norm" in normalized:
        return "model.norm", "norm", "norm", 100
    return normalized, shorten(normalized, 24), "misc", 20_000


def build_layer_groups(gradient_infos: list[Any]) -> tuple[list[dict[str, Any]], list[tuple[str, str]], list[dict[str, Any]]]:
    groups: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for grad in gradient_infos:
        name = str(getattr(grad, "name", "gradient"))
        bytes_value = int(getattr(grad, "bytes", 0) or 0)
        numel = int(getattr(grad, "numel", 0) or 0)
        requires_allreduce = bool(getattr(grad, "requires_tp_allreduce", False))
        group_id, label, kind, order = group_for_gradient(name)
        if group_id not in groups:
            groups[group_id] = {
                "id": group_id,
                "label": label,
                "kind": kind,
                "order": order,
                "count": 0,
                "bytes": 0,
                "numel": 0,
                "allreduce_count": 0,
                "examples": [],
            }
        group = groups[group_id]
        group["count"] += 1
        group["bytes"] += bytes_value
        group["numel"] += numel
        group["allreduce_count"] += int(requires_allreduce)
        if len(group["examples"]) < 3:
            group["examples"].append(name)
        records.append(
            {
                "name": name,
                "group": group_id,
                "shape": list(getattr(grad, "shape", ())),
                "dtype": str(getattr(grad, "dtype", "")),
                "numel": numel,
                "bytes": bytes_value,
                "requires_tp_allreduce": requires_allreduce,
            }
        )
    nodes = sorted(groups.values(), key=lambda item: (item["order"], item["label"]))
    edges = [(nodes[i]["id"], nodes[i + 1]["id"]) for i in range(len(nodes) - 1)]
    return nodes, edges, records


def render_layer_svg(nodes: list[dict[str, Any]], edges: list[tuple[str, str]], output_path: Path, title: str) -> None:
    node_width = 260
    node_height = 68
    gap_y = 86
    width = 1180
    height = max(260, 130 + len(nodes) * gap_y + 90)
    positions: dict[str, tuple[int, int]] = {}
    for idx, node in enumerate(nodes):
        if node["kind"] in {"lm_head", "norm", "embedding"}:
            x = 90
        elif node["kind"] == "layer":
            x = 470
        else:
            x = 820
        positions[node["id"]] = (x, 120 + idx * gap_y)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffaf5"/>',
        f'<text x="36" y="44" font-size="26" font-family="monospace" fill="#7c2d12">{escape(title)}</text>',
        '<text x="36" y="70" font-size="14" font-family="monospace" fill="#9a3412">Real training backward graph, grouped by transformer layer and gradient scope.</text>',
    ]
    for src, dst in edges:
        if src not in positions or dst not in positions:
            continue
        sx, sy = positions[src]
        dx, dy = positions[dst]
        x1 = sx + node_width
        y1 = sy + node_height / 2
        x2 = dx
        y2 = dy + node_height / 2
        mid = (x1 + x2) / 2
        svg.append(f'<path d="M {x1} {y1} C {mid} {y1}, {mid} {y2}, {x2} {y2}" stroke="#fdba74" stroke-width="2" fill="none" opacity="0.55"/>')
    for node in nodes:
        x, y = positions[node["id"]]
        fill = COLORS.get(node["kind"], COLORS["misc"])
        mb = float(node["bytes"]) / (1024 * 1024)
        svg.append(f'<rect x="{x}" y="{y}" width="{node_width}" height="{node_height}" rx="10" fill="{fill}" opacity="0.94"/>')
        svg.append(f'<text x="{x+10}" y="{y+19}" font-size="13" font-family="monospace" fill="#fff">{escape(shorten(node["label"], 26))}</text>')
        svg.append(f'<text x="{x+10}" y="{y+38}" font-size="11" font-family="monospace" fill="#fff">grads={node["count"]} allreduce={node["allreduce_count"]}</text>')
        svg.append(f'<text x="{x+10}" y="{y+56}" font-size="11" font-family="monospace" fill="#fff">bytes={mb:.2f} MiB</text>')
    legend_x = 36
    legend_y = height - 42
    for kind, color in COLORS.items():
        svg.append(f'<rect x="{legend_x}" y="{legend_y}" width="14" height="14" rx="3" fill="{color}"/>')
        svg.append(f'<text x="{legend_x+20}" y="{legend_y+12}" font-size="12" font-family="monospace" fill="#334155">{escape(kind)}</text>')
        legend_x += 132
    svg.append("</svg>")
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def write_index(output_dir: Path) -> None:
    (output_dir / "index.html").write_text(
        """<!doctype html><html><head><meta charset='utf-8'><title>Training Graph</title></head>
<body style='font-family:monospace;background:#f8fafc;margin:24px'>
<h1>Training Graph Views</h1>
<object data='backward_layer_graph.svg' type='image/svg+xml' width='100%' height='960'></object>
</body></html>\n""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype)
    model.train().to(device)
    input_ids, attention_mask = prepare_inputs(tokenizer, args.prompt, device)
    backward_info = extract_backward_graph(model, input_ids, attention_mask, model_name=args.model_path)
    gradient_infos = list(getattr(backward_info, "gradient_infos", []) or [])
    nodes, edges, records = build_layer_groups(gradient_infos)
    render_layer_svg(nodes, edges, output_dir / "backward_layer_graph.svg", "backward layer graph")
    (output_dir / "backward_graph_nodes.json").write_text(json.dumps({"nodes": nodes, "edges": edges, "gradients": records}, indent=2, default=str), encoding="utf-8")
    (output_dir / "backward_graph.txt").write_text("\n".join(f"{r['name']} -> {r['group']} bytes={r['bytes']} allreduce={r['requires_tp_allreduce']}" for r in records) + "\n", encoding="utf-8")
    summary = {
        "model_path": args.model_path,
        "prompt": args.prompt,
        "prompt_tokens": int(input_ids.shape[1]),
        "dtype": args.dtype,
        "forward": {
            "node_count": 0,
            "skipped": "forward torch.export is intentionally skipped for training DAG display",
        },
        "backward": {
            "node_count": len(records),
            "layer_group_count": len(nodes),
            "layer_group_edge_count": len(edges),
            "total_gradient_bytes": int(getattr(backward_info, "total_gradient_bytes", 0) or 0),
            "tp_gradient_bytes": int(getattr(backward_info, "tp_gradient_bytes", 0) or 0),
        },
        "gradient_summary": get_gradient_summary(backward_info),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_index(output_dir)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
