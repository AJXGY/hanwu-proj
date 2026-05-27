from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Cambricon inference execution model artifacts"
    )
    parser.add_argument("--infer-repo", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="mlu:0")
    parser.add_argument("--parallel-mode", choices=["single", "tp"], default="single")
    parser.add_argument("--physical-devices", default="0")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--microbatch-count", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--profile-repeat", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def render_svg(nodes: list[dict], edges: list[tuple[str, str]], title: str) -> str:
    width = 1440
    height = 420
    positions = {}
    for index, node in enumerate(nodes):
        positions[node["id"]] = (140 + index * 240, 210 if node.get("lane", 0) == 0 else 120 + node["lane"] * 110)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="40" y="42" font-size="28" font-family="sans-serif" fill="#111827">{title}</text>',
    ]
    for src, dst in edges:
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        lines.append(
            f'<line x1="{x1 + 90}" y1="{y1}" x2="{x2 - 90}" y2="{y2}" stroke="#64748b" stroke-width="3" marker-end="url(#arrow)"/>'
        )
    lines.insert(
        2,
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#64748b"/></marker></defs>',
    )
    for node in nodes:
        x, y = positions[node["id"]]
        fill = "#dbeafe" if node.get("kind") == "compute" else "#fee2e2" if node.get("kind") == "comm" else "#ecfccb"
        lines.extend(
            [
                f'<rect x="{x - 90}" y="{y - 34}" width="180" height="68" rx="18" fill="{fill}" stroke="#1f2937" stroke-width="2"/>',
                f'<text x="{x}" y="{y - 6}" text-anchor="middle" font-size="18" font-family="sans-serif" fill="#111827">{node["label"]}</text>',
                f'<text x="{x}" y="{y + 18}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#475569">{node.get("sub", "")}</text>',
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    infer_repo = Path(args.infer_repo).expanduser().resolve()
    graph_dir = output_dir / "graph_viz"
    graph_dir.mkdir(parents=True, exist_ok=True)

    export_command = [
        "python3",
        str(infer_repo / "export_graph_viz.py"),
        "--model-path",
        args.model_path,
        "--prompt",
        args.prompt,
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--warmup",
        str(args.warmup),
        "--profile-repeat",
        str(args.profile_repeat),
        "--output-dir",
        str(graph_dir),
    ]
    completed = subprocess.run(
        export_command,
        cwd=str(infer_repo),
        capture_output=True,
        text=True,
        check=False,
    )
    (output_dir / "stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (output_dir / "stderr.log").write_text(completed.stderr or "", encoding="utf-8")

    if completed.returncode != 0:
        summary = {
            "task": "inference_model_structure",
            "success": False,
            "error": "export_graph_viz failed",
            "command": export_command,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    graph_summary = json.loads((graph_dir / "summary.json").read_text(encoding="utf-8"))
    physical_devices = [int(item.strip()) for item in args.physical_devices.split(",") if item.strip()]
    resource_mapping = {
        "cpu_roles": [
            "tokenization / prompt preprocessing",
            "launch orchestration",
            "result postprocessing",
        ],
        "npu_roles": [
            {
                "rank": idx,
                "physical_device": device_id,
                "assignment": (
                    "full model execution"
                    if args.parallel_mode == "single"
                    else "tensor-parallel shard of the transformer layers"
                ),
            }
            for idx, device_id in enumerate(physical_devices)
        ],
    }
    partitioning = {
        "parallel_mode": args.parallel_mode,
        "world_size": args.world_size,
        "tp_size": args.tp_size,
        "multi_card_relation": (
            "single-card execution"
            if args.parallel_mode == "single"
            else "two-card tensor-parallel sharding with collective synchronization"
        ),
    }
    execution_logic = {
        "prefill": "CPU prepares tokens, NPU executes one full prefill pass over all prompt tokens.",
        "decode": "NPU iteratively generates one token per decode step until max_new_tokens is reached.",
        "microbatch_logic": (
            "single request path; no extra microbatch split"
            if args.microbatch_count <= 1
            else f"logical request stream split into {args.microbatch_count} microbatches"
        ),
    }
    nodes = [
        {"id": "cpu_pre", "label": "CPU 预处理", "sub": "tokenize / dispatch", "lane": 1, "kind": "cpu"},
        {"id": "prefill", "label": "NPU Prefill", "sub": f"{graph_summary['prefill']['node_count']} nodes", "lane": 0, "kind": "compute"},
    ]
    edges = [("cpu_pre", "prefill")]
    if args.parallel_mode == "tp":
        nodes.append({"id": "tp_prefill", "label": "TP 通信", "sub": "all-reduce / sync", "lane": 1, "kind": "comm"})
        edges.append(("prefill", "tp_prefill"))
        pre_decode_src = "tp_prefill"
    else:
        pre_decode_src = "prefill"
    nodes.append({"id": "decode", "label": "NPU Decode", "sub": f"{graph_summary['decode']['node_count']} nodes", "lane": 0, "kind": "compute"})
    edges.append((pre_decode_src, "decode"))
    if args.parallel_mode == "tp":
        nodes.append({"id": "tp_decode", "label": "TP 通信", "sub": "decode-step sync", "lane": 1, "kind": "comm"})
        edges.append(("decode", "tp_decode"))
        post_src = "tp_decode"
    else:
        post_src = "decode"
    nodes.append({"id": "cpu_post", "label": "CPU 后处理", "sub": "decode text / return", "lane": 1, "kind": "cpu"})
    edges.append((post_src, "cpu_post"))

    dag = {"nodes": nodes, "edges": edges}
    execution_model = {
        "task": "inference_model_structure",
        "model_path": args.model_path,
        "prompt": args.prompt,
        "prompt_tokens": graph_summary["prompt_tokens"],
        "dtype": args.dtype,
        "resource_mapping": resource_mapping,
        "partitioning": partitioning,
        "execution_logic": execution_logic,
        "graph_summary": graph_summary,
        "dag": dag,
        "artifacts": {
            "graph_viz_dir": str(graph_dir),
            "dag_svg": str(output_dir / "execution_dag.svg"),
            "dag_json": str(output_dir / "execution_dag.json"),
        },
    }
    (output_dir / "execution_model.json").write_text(
        json.dumps(execution_model, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "execution_dag.json").write_text(
        json.dumps(dag, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "execution_dag.svg").write_text(
        render_svg(nodes, edges, "Inference Execution DAG"),
        encoding="utf-8",
    )
    index_html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/><title>Inference Model Structure</title></head>
<body>
<h1>Inference Model Structure</h1>
<p><a href="execution_model.json">execution_model.json</a></p>
<p><a href="execution_dag.svg">execution_dag.svg</a></p>
<p><a href="graph_viz/index.html">graph_viz/index.html</a></p>
<img src="execution_dag.svg" style="max-width:100%;border:1px solid #ddd"/>
</body></html>"""
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")
    summary = {
        "task": "inference_model_structure",
        "success": True,
        "parallel_mode": args.parallel_mode,
        "physical_devices": physical_devices,
        "prompt_tokens": graph_summary["prompt_tokens"],
        "graph_viz_dir": str(graph_dir),
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
