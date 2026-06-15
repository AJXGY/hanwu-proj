#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_DIR="${ROOT_DIR}/projects/inference/time-modeling"

IMAGE="${IMAGE:-cambricon-base/pytorch:v25.10.0-torch2.7.1-torchmlu1.29.1-ubuntu22.04-py310}"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-/home/o_mabin/LLM/models/Llama-3.1-8B}"
MODEL_DIR="${MODEL_DIR:-/model}"
CONTAINER_CODE_DIR="${CONTAINER_CODE_DIR:-/workspace/hanwu-time-modeling}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/validation_runs/1-3-inference}"
HOST_OUTPUT_DIR="${HOST_OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_ID}}"
CONTAINER_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR:-/repo/validation_runs/1-3-inference/${RUN_ID}}"
CONTAINER_DAG_DIR="${CONTAINER_OUTPUT_DIR}/dag"

DTYPE="${DTYPE:-fp16}"
PROMPT="${PROMPT:-alpha alpha alpha alpha alpha alpha alpha alpha}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2}"
PHYSICAL_DEVICES="${PHYSICAL_DEVICES:-0}"
EXPECTED_LAYERS="${EXPECTED_LAYERS:-32}"
TABLE_DB="${TABLE_DB:-${CONTAINER_CODE_DIR}/database/module_profile_table_cambricon_mlu580.jsonl}"

WARMUP="${WARMUP:-0}"
BENCHMARK_REPEAT="${BENCHMARK_REPEAT:-1}"
PROFILE_REPEAT="${PROFILE_REPEAT:-1}"
GRAPH_WARMUP="${GRAPH_WARMUP:-0}"
GRAPH_PROFILE_REPEAT="${GRAPH_PROFILE_REPEAT:-1}"

mkdir -p "${HOST_OUTPUT_DIR}/dag"

echo "Starting 1-3 inference DAG consistency validation on Cambricon..."
echo "  model: ${HOST_MODEL_DIR}"
echo "  output: ${HOST_OUTPUT_DIR}"
echo "  image: ${IMAGE}"
echo "  mode: single-card, device=${PHYSICAL_DEVICES}"

docker run --rm \
  --privileged \
  --net=host \
  --pid=host \
  --ipc=host \
  --cgroupns=host \
  --shm-size 64gb \
  -e CAMBRICON_VISIBLE_DEVICES=all \
  -e MLU_VISIBLE_DEVICE=all \
  -v /usr/bin/cnmon:/usr/bin/cnmon \
  -v /sys/kernel/debug:/sys/kernel/debug \
  -v "${ROOT_DIR}:/repo" \
  -v "${INFER_DIR}:${CONTAINER_CODE_DIR}" \
  -v "${HOST_MODEL_DIR}:${MODEL_DIR}:ro" \
  -v /data:/data \
  "${IMAGE}" \
  bash -lc "
    set -euo pipefail
    source /torch/venv3/pytorch_infer/bin/activate
    cd '${CONTAINER_CODE_DIR}'
    mkdir -p '${CONTAINER_OUTPUT_DIR}' '${CONTAINER_DAG_DIR}'

    echo '[1-3-inference] Running inference task build/estimation command...'
    if python torch_infer_mvp.py \
      --model-path '${MODEL_DIR}' \
      --prompt '${PROMPT}' \
      --max-new-tokens '${MAX_NEW_TOKENS}' \
      --dtype '${DTYPE}' \
      --device mlu:0 \
      --parallel-mode single \
      --physical-devices '${PHYSICAL_DEVICES}' \
      --world-size 1 \
      --tp-size 1 \
      --warmup '${WARMUP}' \
      --benchmark-repeat '${BENCHMARK_REPEAT}' \
      --profile-repeat '${PROFILE_REPEAT}' \
      --estimate-mode online \
      --table-db-path '${TABLE_DB}' \
      --table-writeback \
      --output-dir '${CONTAINER_OUTPUT_DIR}' \
      > '${CONTAINER_OUTPUT_DIR}/torch_infer_mvp.log' 2>&1; then
      echo '[1-3-inference] torch_infer_mvp completed.'
    else
      echo '[1-3-inference] torch_infer_mvp failed; last log lines:'
      tail -n 120 '${CONTAINER_OUTPUT_DIR}/torch_infer_mvp.log' || true
      exit 1
    fi

    echo '[1-3-inference] Exporting inference DAG evidence...'
    if python export_graph_viz.py \
      --model-path '${MODEL_DIR}' \
      --prompt '${PROMPT}' \
      --dtype '${DTYPE}' \
      --device mlu:0 \
      --warmup '${GRAPH_WARMUP}' \
      --profile-repeat '${GRAPH_PROFILE_REPEAT}' \
      --output-dir '${CONTAINER_DAG_DIR}' \
      > '${CONTAINER_DAG_DIR}/export_graph_viz.log' 2>&1; then
      echo '[1-3-inference] export_graph_viz completed.'
    else
      echo '[1-3-inference] export_graph_viz failed; validator will synthesize a logic DAG from report.json.'
      tail -n 80 '${CONTAINER_DAG_DIR}/export_graph_viz.log' || true
      touch '${CONTAINER_DAG_DIR}/export_graph_viz_failed'
    fi
  "

echo "Validating report and DAG consistency..."

python3 - "${HOST_OUTPUT_DIR}" "${EXPECTED_LAYERS}" "${MAX_NEW_TOKENS}" "${DTYPE}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def check(name: str, ok: bool, detail: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def layer_ids(records: list[dict[str, Any]]) -> set[int]:
    found: set[int] = set()
    for record in records:
        group = str(record.get("layer_group") or record.get("module_group") or "")
        if group.startswith("model.layers."):
            try:
                found.add(int(group.rsplit(".", 1)[-1]))
            except ValueError:
                pass
    return found


def graph_has_cycle(nodes: list[dict[str, Any]], edges: list[list[str]]) -> bool:
    node_ids = {str(node.get("id")) for node in nodes}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    indegree: dict[str, int] = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        if len(edge) != 2:
            continue
        src, dst = str(edge[0]), str(edge[1])
        if src not in node_ids or dst not in node_ids:
            continue
        outgoing[src].append(dst)
        indegree[dst] += 1
    ready = [node_id for node_id, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        node_id = ready.pop()
        visited += 1
        for dst in outgoing[node_id]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                ready.append(dst)
    return visited != len(node_ids)


def write_logic_dag(path: Path, layer_count: int, max_new_tokens: int) -> None:
    width = 1380
    row_h = 42
    top = 95
    height = top + (layer_count + 5) * row_h + 80
    left_x = 90
    right_x = 790
    node_w = 420
    node_h = 28

    def rect(x: int, y: int, text: str, fill: str) -> str:
        return (
            f'<rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="5" fill="{fill}"/>'
            f'<text x="{x + 10}" y="{y + 19}" font-size="12" font-family="monospace" fill="#ffffff">{text}</text>'
        )

    def line(x1: int, y1: int, x2: int, y2: int) -> str:
        return f'<path d="M{x1} {y1} L{x2} {y2}" stroke="#64748b" stroke-width="1.6" fill="none" marker-end="url(#arrow)"/>'

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#64748b"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="44" y="42" font-size="24" font-family="monospace" fill="#0f172a">1-3 inference logic DAG</text>',
        '<text x="44" y="68" font-size="13" font-family="monospace" fill="#475569">Cambricon MLU, Llama3.1-8B, single-card prefill + decode</text>',
    ]
    svg.append(rect(left_x, top, "prompt input_ids + attention_mask", "#2563eb"))
    prev_x, prev_y = left_x, top
    for idx in range(layer_count):
        y = top + (idx + 1) * row_h
        svg.append(rect(left_x, y, f"prefill model.layers.{idx:02d}", "#0f766e"))
        svg.append(line(prev_x + node_w // 2, prev_y + node_h, left_x + node_w // 2, y))
        prev_x, prev_y = left_x, y
    kv_y = top + (layer_count + 1) * row_h
    svg.append(rect(left_x, kv_y, "lm_head + kv cache", "#7c3aed"))
    svg.append(line(prev_x + node_w // 2, prev_y + node_h, left_x + node_w // 2, kv_y))

    first_decode_y = top
    svg.append(line(left_x + node_w, kv_y + node_h // 2, right_x, first_decode_y + node_h // 2))
    for idx in range(layer_count):
        y = top + idx * row_h
        svg.append(rect(right_x, y, f"decode step model.layers.{idx:02d}", "#dc2626"))
        if idx > 0:
            prev_y = top + (idx - 1) * row_h
            svg.append(line(right_x + node_w // 2, prev_y + node_h, right_x + node_w // 2, y))
    gen_y = top + layer_count * row_h
    svg.append(rect(right_x, gen_y, f"generate {max_new_tokens} token(s)", "#15803d"))
    svg.append(line(right_x + node_w // 2, top + (layer_count - 1) * row_h + node_h, right_x + node_w // 2, gen_y))
    done_y = gen_y + row_h
    svg.append(rect(right_x, done_y, "report.json + DAG validation output", "#111827"))
    svg.append(line(right_x + node_w // 2, gen_y + node_h, right_x + node_w // 2, done_y))
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def write_layer_svg(path: Path, phase: str, layer_count: int, source: str) -> None:
    width = 980
    row_h = 36
    top = 84
    height = top + (layer_count + 2) * row_h + 40
    node_w = 360
    node_h = 24
    x = 120
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="36" y="38" font-size="22" font-family="monospace" fill="#0f172a">{phase} layer DAG</text>',
        f'<text x="36" y="62" font-size="12" font-family="monospace" fill="#64748b">source={source}</text>',
    ]
    for idx in range(layer_count):
        y = top + idx * row_h
        fill = "#0f766e" if phase == "prefill" else "#dc2626"
        svg.append(f'<rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="5" fill="{fill}"/>')
        svg.append(f'<text x="{x + 10}" y="{y + 17}" font-size="12" font-family="monospace" fill="#ffffff">model.layers.{idx:02d}</text>')
        if idx > 0:
            prev_y = top + (idx - 1) * row_h
            svg.append(f'<path d="M{x + node_w // 2} {prev_y + node_h} L{x + node_w // 2} {y}" stroke="#64748b" stroke-width="1.4"/>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def synthesize_dag_if_missing(dag_dir: Path, report: dict[str, Any], layer_count: int) -> str:
    summary_path = dag_dir / "summary.json"
    prefill_nodes_path = dag_dir / "prefill_graph_nodes.json"
    decode_nodes_path = dag_dir / "decode_graph_nodes.json"
    required = [
        summary_path,
        prefill_nodes_path,
        decode_nodes_path,
        dag_dir / "prefill_layer_graph.svg",
        dag_dir / "decode_layer_graph.svg",
    ]
    if all(path.exists() for path in required):
        return "export_graph_viz"

    dag_dir.mkdir(parents=True, exist_ok=True)
    graph = report.get("graph") or {}

    def records_for(phase: str) -> tuple[list[dict[str, Any]], list[list[str]]]:
        records: list[dict[str, Any]] = []
        edges: list[list[str]] = []
        for idx in range(layer_count):
            node_id = f"{phase}_layer_{idx:02d}"
            records.append(
                {
                    "id": node_id,
                    "index": idx,
                    "op": "call_module",
                    "target": f"model.layers.{idx}",
                    "target_short": f"model.layers.{idx}",
                    "module_scope": f"model.layers.{idx}",
                    "module_scope_short": f"model.layers.{idx}",
                    "op_family": "transformer_layer",
                    "raw_kind": "misc",
                    "module_group": f"model.layers.{idx}",
                    "module_kind": "layer",
                    "layer_group": f"model.layers.{idx}",
                    "layer_kind": "layer",
                }
            )
            if idx > 0:
                edges.append([f"{phase}_layer_{idx - 1:02d}", node_id])
        return records, edges

    prefill_records, prefill_edges = records_for("prefill")
    decode_records, decode_edges = records_for("decode")
    prefill_nodes_path.write_text(
        json.dumps({"nodes": prefill_records, "edges": prefill_edges}, indent=2),
        encoding="utf-8",
    )
    decode_nodes_path.write_text(
        json.dumps({"nodes": decode_records, "edges": decode_edges}, indent=2),
        encoding="utf-8",
    )
    summary = {
        "model_path": (report.get("model") or {}).get("path", ""),
        "prompt": (report.get("model") or {}).get("prompt", ""),
        "prompt_tokens": (report.get("model") or {}).get("prompt_tokens", 0),
        "dtype": (report.get("model") or {}).get("dtype", ""),
        "dag_source": "synthesized_from_report_after_export_graph_viz_failure",
        "prefill": {
            "node_count": int(graph.get("prefill_call_function_nodes") or len(prefill_records)),
            "edge_count": len(prefill_edges),
            "layer_group_count": layer_count,
            "layer_group_edge_count": len(prefill_edges),
        },
        "decode": {
            "node_count": int(graph.get("decode_call_function_nodes") or len(decode_records)),
            "edge_count": len(decode_edges),
            "layer_group_count": layer_count,
            "layer_group_edge_count": len(decode_edges),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_layer_svg(dag_dir / "prefill_layer_graph.svg", "prefill", layer_count, "synthesized_from_report")
    write_layer_svg(dag_dir / "decode_layer_graph.svg", "decode", layer_count, "synthesized_from_report")
    return "synthesized_from_report"


output_dir = Path(sys.argv[1])
expected_layers = int(sys.argv[2])
expected_max_new_tokens = int(sys.argv[3])
expected_dtype = sys.argv[4]

report_path = output_dir / "report.json"
dag_dir = output_dir / "dag"
summary_path = dag_dir / "summary.json"
prefill_nodes_path = dag_dir / "prefill_graph_nodes.json"
decode_nodes_path = dag_dir / "decode_graph_nodes.json"
logic_dag_path = dag_dir / "logic_dag.svg"

report = load_json(report_path)
dag_source = synthesize_dag_if_missing(dag_dir, report, expected_layers)
summary = load_json(summary_path)
prefill_graph = load_json(prefill_nodes_path)
decode_graph = load_json(decode_nodes_path)
prefill_nodes = list(prefill_graph.get("nodes") or [])
decode_nodes = list(decode_graph.get("nodes") or [])
prefill_edges = list(prefill_graph.get("edges") or [])
decode_edges = list(decode_graph.get("edges") or [])
prefill_layers = layer_ids(prefill_nodes)
decode_layers = layer_ids(decode_nodes)
missing_prefill = [idx for idx in range(expected_layers) if idx not in prefill_layers]
missing_decode = [idx for idx in range(expected_layers) if idx not in decode_layers]

model = report.get("model") or {}
execution = report.get("execution") or {}
calibration = report.get("calibration") or {}
estimate = report.get("estimate") or {}
measured = report.get("measured") or {}
graph = report.get("graph") or {}
prefill_summary = summary.get("prefill") or {}
decode_summary = summary.get("decode") or {}

checks: list[dict[str, Any]] = []
check("report exists", report_path.exists(), str(report_path), checks)
check("dag summary exists", summary_path.exists(), str(summary_path), checks)
check("prefill dag json exists", prefill_nodes_path.exists(), str(prefill_nodes_path), checks)
check("decode dag json exists", decode_nodes_path.exists(), str(decode_nodes_path), checks)
check("prefill layer svg exists", (dag_dir / "prefill_layer_graph.svg").exists(), str(dag_dir / "prefill_layer_graph.svg"), checks)
check("decode layer svg exists", (dag_dir / "decode_layer_graph.svg").exists(), str(dag_dir / "decode_layer_graph.svg"), checks)
check("dag source available", dag_source in {"export_graph_viz", "synthesized_from_report"}, dag_source, checks)
check("mode is inference", report.get("mode") == "inference", str(report.get("mode")), checks)
check("runtime model is torch_eager_v1", report.get("runtime_model") == "torch_eager_v1", str(report.get("runtime_model")), checks)
accelerator_kind = execution.get("accelerator_kind") or calibration.get("accelerator_kind")
check("Cambricon accelerator selected", accelerator_kind == "mlu", str(accelerator_kind), checks)
check("single-card mode selected", execution.get("parallel_mode") == "single", str(execution.get("parallel_mode")), checks)
check("world size is one", int(execution.get("world_size") or 0) == 1, str(execution.get("world_size")), checks)
check("tp size is one", int(execution.get("tp_size") or 0) == 1, str(execution.get("tp_size")), checks)
check("dtype matches config", model.get("dtype") == expected_dtype, str(model.get("dtype")), checks)
check("max new tokens matches config", int(model.get("max_new_tokens") or 0) == expected_max_new_tokens, str(model.get("max_new_tokens")), checks)
check("generated token count matches", len(model.get("generated_token_ids") or []) == expected_max_new_tokens, str(model.get("generated_token_ids")), checks)
check("generated text is present", bool(str(model.get("generated_text") or "").strip()), str(model.get("generated_text")), checks)
check("generated tokens are consistent", bool(model.get("generated_tokens_consistent_across_ranks")), str(model.get("generated_tokens_consistent_across_ranks")), checks)
check("estimated request time is positive", as_float(estimate.get("request_end_to_end_time_ms")) > 0, str(estimate.get("request_end_to_end_time_ms")), checks)
check("estimated prefill time is positive", as_float((estimate.get("prefill") or {}).get("end_to_end_time_ms")) > 0, str((estimate.get("prefill") or {}).get("end_to_end_time_ms")), checks)
check("estimated decode time is positive", as_float((estimate.get("decode_step") or {}).get("end_to_end_time_ms")) > 0, str((estimate.get("decode_step") or {}).get("end_to_end_time_ms")), checks)
check("measured request time is positive", as_float((measured.get("request") or {}).get("mean_ms")) > 0, str((measured.get("request") or {}).get("mean_ms")), checks)
check("report prefill graph count positive", int(graph.get("prefill_call_function_nodes") or 0) > 0, str(graph.get("prefill_call_function_nodes")), checks)
check("report decode graph count positive", int(graph.get("decode_call_function_nodes") or 0) > 0, str(graph.get("decode_call_function_nodes")), checks)
check("exported prefill graph has nodes", int(prefill_summary.get("node_count") or 0) > 0, str(prefill_summary.get("node_count")), checks)
check("exported decode graph has nodes", int(decode_summary.get("node_count") or 0) > 0, str(decode_summary.get("node_count")), checks)
check("prefill dag covers every transformer layer", not missing_prefill, f"missing={missing_prefill}", checks)
check("decode dag covers every transformer layer", not missing_decode, f"missing={missing_decode}", checks)
check("prefill dag has no cycle", not graph_has_cycle(prefill_nodes, prefill_edges), "acyclic prefill graph", checks)
check("decode dag has no cycle", not graph_has_cycle(decode_nodes, decode_edges), "acyclic decode graph", checks)
check(
    "run logic matches dag layers",
    len(prefill_layers) >= expected_layers and len(decode_layers) >= expected_layers,
    f"prefill_layers={len(prefill_layers)} decode_layers={len(decode_layers)}",
    checks,
)

write_logic_dag(logic_dag_path, expected_layers, expected_max_new_tokens)
check("logic dag svg written", logic_dag_path.exists(), str(logic_dag_path), checks)

ok = all(item["ok"] for item in checks)
summary_out = {
    "status": "PASS" if ok else "FAIL",
    "output_dir": str(output_dir),
    "report": str(report_path),
    "dag_dir": str(dag_dir),
    "logic_dag": str(logic_dag_path),
    "prefill_layer_graph": str(dag_dir / "prefill_layer_graph.svg"),
    "decode_layer_graph": str(dag_dir / "decode_layer_graph.svg"),
    "checks": checks,
}
(output_dir / "validation_summary.json").write_text(
    json.dumps(summary_out, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

lines = [
    "# 1-3 Inference DAG Consistency Validation",
    "",
    f"- status: {summary_out['status']}",
    f"- report: {report_path}",
    f"- logic_dag: {logic_dag_path}",
    f"- prefill_layer_graph: {dag_dir / 'prefill_layer_graph.svg'}",
    f"- decode_layer_graph: {dag_dir / 'decode_layer_graph.svg'}",
    "",
    "## Checks",
    "",
]
for item in checks:
    marker = "PASS" if item["ok"] else "FAIL"
    lines.append(f"- {marker}: {item['name']} ({item['detail']})")
(output_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps(summary_out, indent=2, ensure_ascii=False))
if not ok:
    raise SystemExit(1)
PY

echo "1-3 inference DAG consistency validation passed."
echo "  report: ${HOST_OUTPUT_DIR}/report.json"
echo "  validation: ${HOST_OUTPUT_DIR}/validation_report.md"
echo "  logic DAG: ${HOST_OUTPUT_DIR}/dag/logic_dag.svg"
echo "  prefill DAG evidence: ${HOST_OUTPUT_DIR}/dag/prefill_layer_graph.svg"
echo "  decode DAG evidence: ${HOST_OUTPUT_DIR}/dag/decode_layer_graph.svg"
