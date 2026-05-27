from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mvp_backend import default_device_string, detect_accelerator_kind
from mvp_calibration import build_calibration
from mvp_estimator import estimate_node, finalize_estimate_ordinals
from mvp_execution import dtype_from_name
from mvp_graph import module_scope_from_stack, op_family_from_target
from mvp_profile import collect_module_profiles
from mvp_runtime import (
    extract_inference_graphs,
    prepare_inputs,
)


RAW_COLORS = {
    "placeholder": "#1d4ed8",
    "embedding": "#0f766e",
    "gemm": "#b45309",
    "attention": "#dc2626",
    "pointwise": "#7c3aed",
    "reduction": "#2563eb",
    "view": "#475569",
    "concat": "#15803d",
    "misc": "#334155",
    "output": "#111827",
}

GROUP_COLORS = {
    "input": "#1d4ed8",
    "embedding": "#0f766e",
    "norm": "#7c3aed",
    "self_attn": "#b45309",
    "mlp": "#be123c",
    "lm_head": "#15803d",
    "misc": "#475569",
    "output": "#111827",
}

ESTIMATE_COLORS = {
    "module_profile_ms": "#dc2626",
    "compute_bound_ms": "#2563eb",
    "memory_bound_ms": "#059669",
    "overhead_ms": "#d97706",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export multi-level torch graph views")
    parser.add_argument("--model-path", default="/opt/models/Llama-3.2-1B")
    parser.add_argument(
        "--prompt",
        default="Explain what a torch-based runtime estimator needs to measure.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="fp16")
    parser.add_argument("--device", default=default_device_string())
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--profile-repeat", type=int, default=10)
    parser.add_argument("--output-dir", default="reports/graph_viz")
    return parser.parse_args()


def escape(value: str) -> str:
    return html.escape(value, quote=True)


def iter_input_nodes(value: Any) -> list[torch.fx.Node]:
    nodes: list[torch.fx.Node] = []
    if isinstance(value, torch.fx.Node):
        nodes.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            nodes.extend(iter_input_nodes(item))
    elif isinstance(value, dict):
        for item in value.values():
            nodes.extend(iter_input_nodes(item))
    return nodes


def shorten(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(limit - 3, 1)] + "..."


def shorten_target(target: Any) -> str:
    return shorten(str(target), 56)


def normalize_scope(scope: str) -> str:
    while scope.startswith("model.model."):
        scope = "model." + scope[len("model.model.") :]
    return scope


def collapse_module_scope(scope: str, node_op: str) -> tuple[str, str]:
    scope = normalize_scope(scope)
    if node_op == "placeholder":
        return "inputs", "input"
    if node_op == "output":
        return "output", "output"
    if scope == "global":
        return "global", "misc"
    parts = scope.split(".")
    if "embed_tokens" in parts:
        return "embed_tokens", "embedding"
    if "lm_head" in parts:
        return "lm_head", "lm_head"
    if parts[-1] == "self_attn":
        return scope, "self_attn"
    if parts[-1] == "mlp":
        return scope, "mlp"
    if "norm" in parts[-1] or parts[-1].endswith("layernorm"):
        return scope, "norm"
    if "layers" in parts:
        layer_idx = parts.index("layers")
        if layer_idx + 1 < len(parts):
            return ".".join(parts[: layer_idx + 2]), "misc"
    return scope, "misc"


def collapse_layer_scope(scope: str, node_op: str) -> tuple[str, str]:
    scope = normalize_scope(scope)
    if node_op == "placeholder":
        return "inputs", "input"
    if node_op == "output":
        return "output", "output"
    if scope == "global":
        return "global", "misc"
    parts = scope.split(".")
    if "embed_tokens" in parts:
        return "embed_tokens", "embedding"
    if "lm_head" in parts:
        return "lm_head", "lm_head"
    if parts[-1] == "norm":
        return scope, "norm"
    if "layers" in parts:
        layer_idx = parts.index("layers")
        if layer_idx + 1 < len(parts):
            return ".".join(parts[: layer_idx + 2]), "misc"
    return scope, "misc"


def raw_kind(node_op: str, op_family: str) -> str:
    if node_op == "placeholder":
        return "placeholder"
    if node_op == "output":
        return "output"
    if op_family in RAW_COLORS:
        return op_family
    return "misc"


def build_node_records(
    graph: torch.fx.Graph,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    records: list[dict[str, Any]] = []
    edges: list[tuple[str, str]] = []
    for index, node in enumerate(graph.nodes):
        scope = normalize_scope(
            module_scope_from_stack(node.meta.get("nn_module_stack"))
        )
        family = (
            op_family_from_target(str(node.target))
            if node.op == "call_function"
            else node.op
        )
        module_group, module_kind = collapse_module_scope(scope, node.op)
        layer_group, layer_kind = collapse_layer_scope(scope, node.op)
        records.append(
            {
                "id": node.name,
                "index": index,
                "op": node.op,
                "target": str(node.target),
                "target_short": shorten_target(node.target),
                "module_scope": scope,
                "module_scope_short": shorten(scope, 36),
                "op_family": family,
                "raw_kind": raw_kind(node.op, family),
                "module_group": module_group,
                "module_kind": module_kind,
                "layer_group": layer_group,
                "layer_kind": layer_kind,
            }
        )
        for arg in iter_input_nodes(node.args):
            edges.append((arg.name, node.name))
        for kwarg in iter_input_nodes(node.kwargs):
            edges.append((kwarg.name, node.name))
    return records, edges


def build_collapsed_graph(
    records: list[dict[str, Any]],
    edges: list[tuple[str, str]],
    group_key: str,
    kind_key: str,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    by_id = {record["id"]: record for record in records}
    group_order: list[str] = []
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        group_name = record[group_key]
        if group_name not in groups:
            group_order.append(group_name)
            groups[group_name] = {
                "id": group_name,
                "label": group_name,
                "kind": record[kind_key],
                "count": 0,
                "families": defaultdict(int),
                "ops": defaultdict(int),
                "first_index": record["index"],
            }
        group = groups[group_name]
        group["count"] += 1
        group["families"][record["op_family"]] += 1
        group["ops"][record["op"]] += 1
        group["first_index"] = min(group["first_index"], record["index"])

    collapsed_edges: set[tuple[str, str]] = set()
    for src_id, dst_id in edges:
        src_group = by_id[src_id][group_key]
        dst_group = by_id[dst_id][group_key]
        if src_group != dst_group:
            collapsed_edges.add((src_group, dst_group))

    nodes: list[dict[str, Any]] = []
    for group_name in sorted(group_order, key=lambda item: groups[item]["first_index"]):
        item = groups[group_name]
        families = sorted(item["families"].items(), key=lambda kv: (-kv[1], kv[0]))
        ops = sorted(item["ops"].items(), key=lambda kv: (-kv[1], kv[0]))
        nodes.append(
            {
                "id": item["id"],
                "label": item["label"],
                "kind": item["kind"],
                "count": item["count"],
                "top_family": families[0][0] if families else "misc",
                "top_op": ops[0][0] if ops else "unknown",
                "first_index": item["first_index"],
            }
        )
    return nodes, sorted(collapsed_edges)


def positions_for(
    nodes: list[dict[str, Any]], x_map: dict[str, int], gap_y: int
) -> dict[str, tuple[int, int]]:
    positions: dict[str, tuple[int, int]] = {}
    for idx, node in enumerate(nodes):
        positions[node["id"]] = (
            x_map.get(node["kind"], x_map["misc"]),
            90 + idx * gap_y,
        )
    return positions


def render_svg(
    *,
    title: str,
    subtitle: str,
    nodes: list[dict[str, Any]],
    edges: list[tuple[str, str]],
    output_path: Path,
    colors: dict[str, str],
    x_map: dict[str, int],
    node_width: int,
    node_height: int,
    gap_y: int,
    label_lines: list[str],
) -> None:
    width = max(x_map.values()) + node_width + 160
    height = 90 + len(nodes) * gap_y + 120
    positions = positions_for(nodes, x_map, gap_y)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="36" y="42" font-size="26" font-family="monospace" fill="#0f172a">{escape(title)}</text>',
        f'<text x="36" y="66" font-size="14" font-family="monospace" fill="#475569">{escape(subtitle)}</text>',
    ]

    for src_id, dst_id in edges:
        if src_id not in positions or dst_id not in positions:
            continue
        sx, sy = positions[src_id]
        dx, dy = positions[dst_id]
        x1 = sx + node_width
        y1 = sy + node_height / 2
        x2 = dx
        y2 = dy + node_height / 2
        mid_x = (x1 + x2) / 2
        svg.append(
            "".join(
                [
                    f'<path d="M {x1} {y1} C {mid_x} {y1}, {mid_x} {y2}, {x2} {y2}" ',
                    'stroke="#94a3b8" stroke-width="1" fill="none" opacity="0.45"/>',
                ]
            )
        )

    for node in nodes:
        x, y = positions[node["id"]]
        fill = colors.get(node["kind"], colors["misc"])
        svg.append(
            f'<rect x="{x}" y="{y}" width="{node_width}" height="{node_height}" rx="10" fill="{fill}" opacity="0.94"/>'
        )
        for line_idx, key in enumerate(label_lines):
            font = 12 if line_idx == 0 else 10
            value = str(node.get(key, ""))
            svg.append(
                f'<text x="{x + 10}" y="{y + 17 + line_idx * 13}" font-size="{font}" font-family="monospace" fill="#ffffff">{escape(value)}</text>'
            )

    legend_x = 36
    legend_y = height - 44
    for kind, fill in colors.items():
        svg.append(
            f'<rect x="{legend_x}" y="{legend_y}" width="14" height="14" rx="3" fill="{fill}"/>'
        )
        svg.append(
            f'<text x="{legend_x + 20}" y="{legend_y + 12}" font-size="12" font-family="monospace" fill="#334155">{escape(kind)}</text>'
        )
        legend_x += 128

    svg.append("</svg>")
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def render_raw_graph(
    name: str,
    records: list[dict[str, Any]],
    edges: list[tuple[str, str]],
    output_dir: Path,
) -> None:
    nodes = []
    for record in records:
        nodes.append(
            {
                "id": record["id"],
                "kind": record["raw_kind"],
                "line1": shorten(record["id"], 24),
                "line2": shorten(record["target_short"], 28),
                "line3": shorten(record["module_scope_short"], 28),
            }
        )
    render_svg(
        title=f"{name} raw graph",
        subtitle="Level 0: every exported FX node in execution order.",
        nodes=nodes,
        edges=edges,
        output_path=output_dir / f"{name}_raw_graph.svg",
        colors=RAW_COLORS,
        x_map={
            "placeholder": 50,
            "embedding": 220,
            "gemm": 390,
            "attention": 560,
            "pointwise": 730,
            "reduction": 900,
            "view": 1070,
            "concat": 1240,
            "misc": 1410,
            "output": 1580,
        },
        node_width=150,
        node_height=40,
        gap_y=18,
        label_lines=["line1", "line2", "line3"],
    )


def render_group_graph(
    name: str,
    nodes: list[dict[str, Any]],
    edges: list[tuple[str, str]],
    output_dir: Path,
    suffix: str,
    subtitle: str,
) -> None:
    svg_nodes = []
    for node in nodes:
        svg_nodes.append(
            {
                "id": node["id"],
                "kind": node["kind"],
                "line1": shorten(node["label"], 28),
                "line2": f"nodes={node['count']} top={node['top_family']}",
                "line3": f"op={node['top_op']}",
            }
        )
    render_svg(
        title=f"{name} {suffix}",
        subtitle=subtitle,
        nodes=svg_nodes,
        edges=edges,
        output_path=output_dir / f"{name}_{suffix}.svg",
        colors=GROUP_COLORS,
        x_map={
            "input": 70,
            "embedding": 280,
            "norm": 500,
            "self_attn": 740,
            "mlp": 980,
            "lm_head": 1220,
            "misc": 500,
            "output": 1460,
        },
        node_width=210,
        node_height=48,
        gap_y=54,
        label_lines=["line1", "line2", "line3"],
    )


def build_estimate_groups(
    phase_name: str,
    estimate_records,
    module_profiles,
) -> dict[str, dict[str, float]]:
    by_group: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "module_profile_ms": 0.0,
            "compute_bound_ms": 0.0,
            "memory_bound_ms": 0.0,
            "overhead_ms": 0.0,
            "total_ms": 0.0,
        }
    )
    covered = {
        node_id for record in module_profiles for node_id in record.covered_node_ids
    }

    for estimate in estimate_records:
        if estimate.node_name in covered:
            continue
        group_name, _ = collapse_layer_scope(estimate.module_scope, "call_function")
        if estimate.compute_time_ms >= estimate.memory_time_ms:
            by_group[group_name]["compute_bound_ms"] += estimate.compute_time_ms
        else:
            by_group[group_name]["memory_bound_ms"] += estimate.memory_time_ms
        by_group[group_name]["overhead_ms"] += estimate.runtime_overhead_ms
        by_group[group_name]["total_ms"] += estimate.estimated_time_ms

    for record in module_profiles:
        group_name, _ = collapse_layer_scope(record.module_scope, phase_name)
        by_group[group_name]["module_profile_ms"] += record.mean_ms
        by_group[group_name]["total_ms"] += record.mean_ms

    return by_group


def parse_layer_index(label: str) -> int | None:
    if ".layers." not in label:
        return None
    suffix = label.split(".layers.", 1)[1]
    digits = []
    for ch in suffix:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        return None
    return int("".join(digits))


def estimate_role(label: str) -> tuple[str, int]:
    if label == "inputs":
        return "backbone", -100
    if label == "embed_tokens":
        return "backbone", -90
    layer_idx = parse_layer_index(label)
    if layer_idx is not None:
        return "backbone", layer_idx
    if label.endswith("norm") or ".norm" in label or label == "model.norm":
        return "backbone", 10_000
    if "lm_head" in label:
        return "backbone", 10_100
    if label == "output":
        return "backbone", 10_200
    return "support", 20_000


def display_label(label: str) -> str:
    layer_idx = parse_layer_index(label)
    if layer_idx is not None:
        return f"layer {layer_idx:02d}"
    if label == "embed_tokens":
        return "embed"
    return label


def render_estimate_graph(
    name: str,
    layer_nodes: list[dict[str, Any]],
    layer_edges: list[tuple[str, str]],
    estimate_groups: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    width = 1580
    node_width = 300
    node_height = 74
    gap_y = 86
    support_nodes = []
    backbone_nodes = []
    for node in layer_nodes:
        role, order = estimate_role(node["label"])
        item = dict(node)
        item["role"] = role
        item["order"] = order
        if role == "backbone":
            backbone_nodes.append(item)
        else:
            support_nodes.append(item)
    backbone_nodes.sort(key=lambda item: item["order"])
    support_nodes.sort(key=lambda item: item["label"])
    ordered_nodes = support_nodes + backbone_nodes
    height = 120 + max(len(backbone_nodes) * gap_y, len(support_nodes) * 98) + 140

    positions: dict[str, tuple[int, int]] = {}
    for idx, node in enumerate(support_nodes):
        positions[node["id"]] = (60, 120 + idx * 98)
    for idx, node in enumerate(backbone_nodes):
        positions[node["id"]] = (620, 120 + idx * gap_y)

    backbone_edges = []
    for idx in range(len(backbone_nodes) - 1):
        backbone_edges.append(
            (backbone_nodes[idx]["id"], backbone_nodes[idx + 1]["id"])
        )

    phase_total = sum(
        estimate_groups.get(node["id"], {}).get("total_ms", 0.0) for node in layer_nodes
    )
    max_total = max(
        [
            estimate_groups.get(node["id"], {}).get("total_ms", 0.0)
            for node in layer_nodes
        ]
        + [1.0]
    )

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffaf5"/>',
        f'<text x="36" y="42" font-size="26" font-family="monospace" fill="#7c2d12">{escape(name + " estimate graph")}</text>',
        '<text x="36" y="66" font-size="14" font-family="monospace" fill="#9a3412">Level 3: backbone-first view. Bold chain shows execution order; left boxes are shared support groups detached to reduce clutter.</text>',
        f'<text x="1040" y="42" font-size="18" font-family="monospace" fill="#7c2d12">phase_total={phase_total:.4f} ms</text>',
    ]

    if backbone_nodes:
        x = 770
        y1 = positions[backbone_nodes[0]["id"]][1] + node_height / 2
        y2 = positions[backbone_nodes[-1]["id"]][1] + node_height / 2
        svg.append(
            f'<path d="M {x} {y1} L {x} {y2}" stroke="#fdba74" stroke-width="4" fill="none" opacity="0.65"/>'
        )

    for src_id, dst_id in backbone_edges:
        if src_id not in positions or dst_id not in positions:
            continue
        sx, sy = positions[src_id]
        dx, dy = positions[dst_id]
        x1 = sx + node_width
        y1 = sy + node_height / 2
        x2 = dx
        y2 = dy + node_height / 2
        mid_x = (x1 + x2) / 2
        svg.append(
            "".join(
                [
                    f'<path d="M {x1} {y1} C {mid_x} {y1}, {mid_x} {y2}, {x2} {y2}" ',
                    'stroke="#9a3412" stroke-width="2.2" fill="none" opacity="0.85"/>',
                ]
            )
        )

    for node in ordered_nodes:
        x, y = positions[node["id"]]
        comps = estimate_groups.get(
            node["id"],
            {
                "module_profile_ms": 0.0,
                "compute_bound_ms": 0.0,
                "memory_bound_ms": 0.0,
                "overhead_ms": 0.0,
                "total_ms": 0.0,
            },
        )
        total_ms = comps["total_ms"]
        intensity = min(1.0, total_ms / max_total)
        fill = f"rgba(180, 83, 9, {0.20 + 0.55 * intensity:.3f})"
        stroke = "#b45309" if node["role"] == "backbone" else "#c2410c"
        svg.append(
            f'<rect x="{x}" y="{y}" width="{node_width}" height="{node_height}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
        )
        svg.append(
            f'<text x="{x + 12}" y="{y + 18}" font-size="12" font-family="monospace" fill="#431407">{escape(shorten(display_label(node["label"]), 34))}</text>'
        )
        svg.append(
            f'<text x="{x + 12}" y="{y + 33}" font-size="11" font-family="monospace" fill="#7c2d12">total={total_ms:.4f} ms</text>'
        )
        svg.append(
            f'<text x="{x + 12}" y="{y + 47}" font-size="10" font-family="monospace" fill="#92400e">prof={comps["module_profile_ms"]:.3f} comp={comps["compute_bound_ms"]:.3f}</text>'
        )
        svg.append(
            f'<text x="{x + 12}" y="{y + 60}" font-size="10" font-family="monospace" fill="#92400e">mem={comps["memory_bound_ms"]:.3f} ovr={comps["overhead_ms"]:.3f}</text>'
        )

        bar_x = x + 12
        bar_y = y + 64
        bar_w = 240
        if total_ms > 0:
            offsets = 0.0
            for key in [
                "module_profile_ms",
                "compute_bound_ms",
                "memory_bound_ms",
                "overhead_ms",
            ]:
                width_part = bar_w * comps[key] / total_ms
                if width_part <= 0:
                    continue
                svg.append(
                    f'<rect x="{bar_x + offsets:.2f}" y="{bar_y}" width="{width_part:.2f}" height="6" rx="2" fill="{ESTIMATE_COLORS[key]}"/>'
                )
                offsets += width_part
        else:
            svg.append(
                f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="6" rx="2" fill="#e2e8f0"/>'
            )

        if node["role"] == "support":
            anchor_x = x + node_width
            anchor_y = y + node_height / 2
            dst_x = 620
            dst_y = (
                positions[backbone_nodes[0]["id"]][1] + node_height / 2
                if backbone_nodes
                else anchor_y
            )
            mid_x = (anchor_x + dst_x) / 2
            svg.append(
                "".join(
                    [
                        f'<path d="M {anchor_x} {anchor_y} C {mid_x} {anchor_y}, {mid_x} {dst_y}, {dst_x} {dst_y}" ',
                        'stroke="#cbd5e1" stroke-width="1.0" stroke-dasharray="4 4" fill="none" opacity="0.55"/>',
                    ]
                )
            )

    legend_x = 36
    legend_y = height - 44
    for key in [
        "module_profile_ms",
        "compute_bound_ms",
        "memory_bound_ms",
        "overhead_ms",
    ]:
        fill = ESTIMATE_COLORS[key]
        svg.append(
            f'<rect x="{legend_x}" y="{legend_y}" width="14" height="14" rx="3" fill="{fill}"/>'
        )
        svg.append(
            f'<text x="{legend_x + 20}" y="{legend_y + 12}" font-size="12" font-family="monospace" fill="#7c2d12">{key}</text>'
        )
        legend_x += 180

    svg.append("</svg>")
    (output_dir / f"{name}_estimate_graph.svg").write_text(
        "\n".join(svg) + "\n", encoding="utf-8"
    )


def build_index_html(output_dir: Path, summary: dict[str, Any]) -> None:
    blocks = []
    for phase in ["prefill", "decode"]:
        blocks.append(
            f"""
<section>
  <h2>{phase}</h2>
  <p>{summary[phase]["node_count"]} raw nodes, {summary[phase]["module_group_count"]} module groups, {summary[phase]["layer_group_count"]} layer groups.</p>
  <ul>
    <li><a href="{phase}_raw_graph.svg">Level 0 raw graph</a></li>
    <li><a href="{phase}_module_graph.svg">Level 1 module graph</a></li>
    <li><a href="{phase}_layer_graph.svg">Level 2 layer graph</a></li>
    <li><a href="{phase}_estimate_graph.svg">Level 3 estimate graph</a></li>
    <li><a href="{phase}_graph.txt">Raw FX graph text</a></li>
    <li><a href="{phase}_graph_nodes.json">Node and edge JSON</a></li>
  </ul>
  <object data="{phase}_layer_graph.svg" type="image/svg+xml" width="100%" height="820"></object>
  <object data="{phase}_estimate_graph.svg" type="image/svg+xml" width="100%" height="980"></object>
</section>
"""
        )
    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Torch Graph Views</title>
  <style>
    body {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 24px; background: #f8fafc; color: #0f172a; }}
    section {{ margin-bottom: 36px; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    ul {{ line-height: 1.7; }}
    object {{ border: 1px solid #e2e8f0; border-radius: 8px; background: white; margin-top: 12px; }}
  </style>
</head>
<body>
  <h1>Torch Graph Views</h1>
  <p>Prompt tokens: {summary["prompt_tokens"]}. Multi-level views: raw FX nodes, module groups, layer groups, and estimate overlays.</p>
  {"".join(blocks)}
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def export_phase(
    *,
    name: str,
    graph: torch.fx.Graph,
    estimate_records,
    module_profiles,
    output_dir: Path,
) -> dict[str, Any]:
    records, edges = build_node_records(graph)
    module_nodes, module_edges = build_collapsed_graph(
        records, edges, "module_group", "module_kind"
    )
    layer_nodes, layer_edges = build_collapsed_graph(
        records, edges, "layer_group", "layer_kind"
    )
    estimate_groups = build_estimate_groups(name, estimate_records, module_profiles)

    (output_dir / f"{name}_graph.txt").write_text(str(graph) + "\n", encoding="utf-8")
    (output_dir / f"{name}_graph_nodes.json").write_text(
        json.dumps({"nodes": records, "edges": edges}, indent=2), encoding="utf-8"
    )

    render_raw_graph(name, records, edges, output_dir)
    render_group_graph(
        name,
        module_nodes,
        module_edges,
        output_dir,
        "module_graph",
        "Level 1: collapse raw nodes by module scope.",
    )
    render_group_graph(
        name,
        layer_nodes,
        layer_edges,
        output_dir,
        "layer_graph",
        "Level 2: collapse module groups into transformer layers.",
    )
    render_estimate_graph(name, layer_nodes, layer_edges, estimate_groups, output_dir)

    return {
        "node_count": len(records),
        "edge_count": len(edges),
        "module_group_count": len(module_nodes),
        "module_group_edge_count": len(module_edges),
        "layer_group_count": len(layer_nodes),
        "layer_group_edge_count": len(layer_edges),
    }


def main() -> None:
    args = parse_args()
    detect_accelerator_kind(args.device)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype)
    model.eval().to(device)

    input_ids, attention_mask = prepare_inputs(tokenizer, args.prompt, device)
    graphs = extract_inference_graphs(model, input_ids, attention_mask)
    calibration = build_calibration(dtype, device)

    prefill_estimates = finalize_estimate_ordinals(
        [
            estimate
            for node in graphs["prefill_export"].graph.nodes
            if (estimate := estimate_node(node, "prefill", calibration)) is not None
        ]
    )
    decode_estimates = finalize_estimate_ordinals(
        [
            estimate
            for node in graphs["decode_export"].graph.nodes
            if (estimate := estimate_node(node, "decode_step", calibration)) is not None
        ]
    )

    module_profiles = collect_module_profiles(
        model=model,
        input_ids=graphs["input_ids"],
        attention_mask=graphs["attention_mask"],
        next_token=graphs["next_token"],
        next_attention_mask=graphs["next_attention_mask"],
        decode_past=graphs["prefill_outputs"].decode_past,
        prefill_estimates=prefill_estimates,
        decode_estimates=decode_estimates,
        warmup=args.warmup,
        repeat=args.profile_repeat,
    )

    summary = {
        "model_path": args.model_path,
        "prompt": args.prompt,
        "prompt_tokens": int(graphs["input_ids"].shape[1]),
        "dtype": args.dtype,
        "prefill": export_phase(
            name="prefill",
            graph=graphs["prefill_export"].graph,
            estimate_records=prefill_estimates,
            module_profiles=module_profiles["prefill"],
            output_dir=output_dir,
        ),
        "decode": export_phase(
            name="decode",
            graph=graphs["decode_export"].graph,
            estimate_records=decode_estimates,
            module_profiles=module_profiles["decode_step"],
            output_dir=output_dir,
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    build_index_html(output_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
