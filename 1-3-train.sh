#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="${ROOT_DIR}/projects/training/time-modeling"

IMAGE="${IMAGE:-cambricon-base/pytorch:v25.10.0-torch2.7.1-torchmlu1.29.1-ubuntu22.04-py310}"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-/home/o_mabin/LLM/models/Llama-3.1-8B}"
MODEL_DIR="${MODEL_DIR:-/model}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/validation_runs/1-3-train}"
HOST_OUTPUT_DIR="${HOST_OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_ID}}"
CONTAINER_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR:-/repo/validation_runs/1-3-train/${RUN_ID}}"
CONTAINER_DAG_DIR="${CONTAINER_OUTPUT_DIR}/dag"

PARALLEL_MODE="${PARALLEL_MODE:-single}"
PHYSICAL_DEVICES="${PHYSICAL_DEVICES:-}"
WORLD_SIZE="${WORLD_SIZE:-}"
TP_SIZE="${TP_SIZE:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29513}"

DTYPE="${DTYPE:-bf16}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
MICROBATCH_COUNT="${MICROBATCH_COUNT:-1}"
MICROBATCH_SIZE="${MICROBATCH_SIZE:-1}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-sgd}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16.0}"
EXPECTED_LAYERS="${EXPECTED_LAYERS:-32}"
GRAPH_PROMPT="${GRAPH_PROMPT:-alpha alpha alpha alpha alpha alpha alpha}"
PROFILE_DB="${PROFILE_DB:-}"

WARMUP="${WARMUP:-1}"
BENCHMARK_REPEAT="${BENCHMARK_REPEAT:-3}"
PROFILE_REPEAT="${PROFILE_REPEAT:-3}"

if [[ "${PARALLEL_MODE}" == "single" ]]; then
  PHYSICAL_DEVICES="${PHYSICAL_DEVICES:-0}"
  WORLD_SIZE="1"
  TP_SIZE="1"
  NPROC_PER_NODE="1"
  PROFILE_DB="${PROFILE_DB:-/workspace/database/train_component_profile_cambricon_mlu580_tp_single.jsonl}"
else
  PHYSICAL_DEVICES="${PHYSICAL_DEVICES:-0,1}"
  WORLD_SIZE="${WORLD_SIZE:-2}"
  TP_SIZE="${TP_SIZE:-2}"
  NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
  PROFILE_DB="${PROFILE_DB:-/workspace/database/train_component_profile_cambricon_mlu580_tp2.jsonl}"
fi

mkdir -p "${HOST_OUTPUT_DIR}/dag"

echo "Starting 1-3 training DAG consistency validation on Cambricon..."
echo "  model: ${HOST_MODEL_DIR}"
echo "  output: ${HOST_OUTPUT_DIR}"
echo "  image: ${IMAGE}"
echo "  mode: ${PARALLEL_MODE}, world_size=${WORLD_SIZE}, tp_size=${TP_SIZE}"

if [[ "${PARALLEL_MODE}" == "tp" && "${WORLD_SIZE}" != "1" ]]; then
  TRAIN_LAUNCH="python -m torch.distributed.run --nproc_per_node '${NPROC_PER_NODE}' --master_addr '${MASTER_ADDR}' --master_port '${MASTER_PORT}' torch_train_tp_mvp.py"
else
  TRAIN_LAUNCH="python torch_train_tp_mvp.py"
fi

docker run --rm \
  --privileged \
  --net=host \
  --pid=host \
  --ipc=host \
  --cgroupns=host \
  --shm-size 64gb \
  -e CAMBRICON_VISIBLE_DEVICES=all \
  -e MLU_VISIBLE_DEVICE=all \
  -e PYTORCH_MLU_ALLOC_CONF=expandable_segments:True \
  -v /usr/bin/cnmon:/usr/bin/cnmon \
  -v /sys/kernel/debug:/sys/kernel/debug \
  -v "${ROOT_DIR}:/repo" \
  -v "${TRAIN_DIR}:/workspace" \
  -v "${HOST_MODEL_DIR}:${MODEL_DIR}:ro" \
  -v /data:/data \
  "${IMAGE}" \
  bash -lc "
    set -euo pipefail
    source /torch/venv3/pytorch/bin/activate
    cd /workspace
    mkdir -p '${CONTAINER_OUTPUT_DIR}' '${CONTAINER_DAG_DIR}'

    echo '[1-3-train] Running training task build/estimation command...'
    if ${TRAIN_LAUNCH} \
      --model-path '${MODEL_DIR}' \
      --dtype '${DTYPE}' \
      --device mlu:0 \
      --parallel-mode '${PARALLEL_MODE}' \
      --physical-devices '${PHYSICAL_DEVICES}' \
      --world-size '${WORLD_SIZE}' \
      --tp-size '${TP_SIZE}' \
      --nproc-per-node '${NPROC_PER_NODE}' \
      --microbatch-count '${MICROBATCH_COUNT}' \
      --microbatch-size '${MICROBATCH_SIZE}' \
      --sequence-length '${SEQUENCE_LENGTH}' \
      --optimizer-type '${OPTIMIZER_TYPE}' \
      --lora-rank '${LORA_RANK}' \
      --lora-alpha '${LORA_ALPHA}' \
      --estimate-mode online \
      --profile-db-path '${PROFILE_DB}' \
      --write-profile-db \
      --warmup '${WARMUP}' \
      --benchmark-repeat '${BENCHMARK_REPEAT}' \
      --profile-repeat '${PROFILE_REPEAT}' \
      --output-dir '${CONTAINER_OUTPUT_DIR}' \
      --enable-gradient-checkpointing \
      > '${CONTAINER_OUTPUT_DIR}/torch_train_tp_mvp.log' 2>&1; then
      echo '[1-3-train] torch_train_tp_mvp completed.'
    else
      echo '[1-3-train] torch_train_tp_mvp failed; last log lines:'
      tail -n 120 '${CONTAINER_OUTPUT_DIR}/torch_train_tp_mvp.log' || true
      exit 1
    fi

    echo '[1-3-train] Exporting training DAG evidence...'
    if python export_train_graph.py \
      --model-path '${MODEL_DIR}' \
      --prompt '${GRAPH_PROMPT}' \
      --dtype '${DTYPE}' \
      --device mlu:0 \
      --output-dir '${CONTAINER_DAG_DIR}' \
      > '${CONTAINER_DAG_DIR}/export_train_graph.log' 2>&1; then
      echo '[1-3-train] export_train_graph completed.'
    else
      echo '[1-3-train] export_train_graph failed; last log lines:'
      tail -n 120 '${CONTAINER_DAG_DIR}/export_train_graph.log' || true
      exit 1
    fi
  "

echo "Validating report and DAG consistency..."

python3 - "${HOST_OUTPUT_DIR}" "${EXPECTED_LAYERS}" "${PARALLEL_MODE}" "${WORLD_SIZE}" "${TP_SIZE}" "${SEQUENCE_LENGTH}" "${DTYPE}" "${OPTIMIZER_TYPE}" "${LORA_RANK}" <<'PY'
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


def layer_ids(nodes: list[dict[str, Any]]) -> set[int]:
    found: set[int] = set()
    for node in nodes:
        node_id = str(node.get("id", ""))
        if node_id.startswith("model.layers."):
            try:
                found.add(int(node_id.rsplit(".", 1)[-1]))
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


def write_logic_dag(path: Path, layer_count: int, parallel_mode: str, tp_size: int) -> None:
    width = 1380
    row_h = 42
    top = 95
    height = top + (layer_count + 4) * row_h + 80
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
        '<text x="44" y="42" font-size="24" font-family="monospace" fill="#0f172a">1-3 training logic DAG</text>',
        f'<text x="44" y="68" font-size="13" font-family="monospace" fill="#475569">Cambricon MLU, Llama3.1-8B, mode={parallel_mode}, tp_size={tp_size}</text>',
    ]
    svg.append(rect(left_x, top, "input_ids + attention_mask", "#2563eb"))
    prev_x, prev_y = left_x, top
    for idx in range(layer_count):
        y = top + (idx + 1) * row_h
        svg.append(rect(left_x, y, f"forward model.layers.{idx:02d}", "#0f766e"))
        svg.append(line(prev_x + node_w // 2, prev_y + node_h, left_x + node_w // 2, y))
        prev_x, prev_y = left_x, y
    loss_y = top + (layer_count + 1) * row_h
    svg.append(rect(left_x, loss_y, "lm_head + loss", "#7c3aed"))
    svg.append(line(prev_x + node_w // 2, prev_y + node_h, left_x + node_w // 2, loss_y))

    first_back_y = top
    svg.append(line(left_x + node_w, loss_y + node_h // 2, right_x, first_back_y + node_h // 2))
    for pos, idx in enumerate(reversed(range(layer_count))):
        y = top + pos * row_h
        suffix = " + TP allreduce" if parallel_mode == "tp" and tp_size > 1 else ""
        svg.append(rect(right_x, y, f"backward model.layers.{idx:02d}{suffix}", "#dc2626"))
        if pos > 0:
            prev_y = top + (pos - 1) * row_h
            svg.append(line(right_x + node_w // 2, prev_y + node_h, right_x + node_w // 2, y))
    opt_y = top + layer_count * row_h
    svg.append(rect(right_x, opt_y, "optimizer step", "#15803d"))
    svg.append(line(right_x + node_w // 2, top + (layer_count - 1) * row_h + node_h, right_x + node_w // 2, opt_y))
    done_y = opt_y + row_h
    svg.append(rect(right_x, done_y, "report.json + DAG validation output", "#111827"))
    svg.append(line(right_x + node_w // 2, opt_y + node_h, right_x + node_w // 2, done_y))
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


output_dir = Path(sys.argv[1])
expected_layers = int(sys.argv[2])
expected_parallel_mode = sys.argv[3]
expected_world_size = int(sys.argv[4])
expected_tp_size = int(sys.argv[5])
expected_seq_len = int(sys.argv[6])
expected_dtype = sys.argv[7]
expected_optimizer = sys.argv[8]
expected_lora_rank = int(sys.argv[9])

report_path = output_dir / "report.json"
dag_dir = output_dir / "dag"
summary_path = dag_dir / "summary.json"
nodes_path = dag_dir / "backward_graph_nodes.json"
logic_dag_path = dag_dir / "logic_dag.svg"

report = load_json(report_path)
summary = load_json(summary_path)
graph_nodes = load_json(nodes_path)
nodes = list(graph_nodes.get("nodes") or [])
edges = list(graph_nodes.get("edges") or [])
layers = layer_ids(nodes)
missing_layers = [idx for idx in range(expected_layers) if idx not in layers]

model = report.get("model") or {}
execution = report.get("execution") or {}
measured = report.get("measured") or {}
estimate = report.get("estimate") or {}
phase_estimates = report.get("phase_estimates") or {}
backward = summary.get("backward") or {}

checks: list[dict[str, Any]] = []
check("report exists", report_path.exists(), str(report_path), checks)
check("dag summary exists", summary_path.exists(), str(summary_path), checks)
check("dag node json exists", nodes_path.exists(), str(nodes_path), checks)
check("dag svg exists", (dag_dir / "backward_layer_graph.svg").exists(), str(dag_dir / "backward_layer_graph.svg"), checks)
check("mode is training", report.get("mode") == "training", str(report.get("mode")), checks)
check("runtime model is torch_tp_train_v1", report.get("runtime_model") == "torch_tp_train_v1", str(report.get("runtime_model")), checks)
check("model type is llama", model.get("model_type") == "llama", str(model.get("model_type")), checks)
check("layer count matches Llama3.1-8B", int(model.get("num_hidden_layers") or 0) == expected_layers, str(model.get("num_hidden_layers")), checks)
check("hidden size matches Llama3.1-8B", int(model.get("hidden_size") or 0) == 4096, str(model.get("hidden_size")), checks)
check("attention heads match Llama3.1-8B", int(model.get("num_attention_heads") or 0) == 32, str(model.get("num_attention_heads")), checks)
check("Cambricon accelerator selected", execution.get("accelerator_kind") == "mlu", str(execution.get("accelerator_kind")), checks)
check("parallel mode matches config", execution.get("parallel_mode") == expected_parallel_mode, str(execution.get("parallel_mode")), checks)
check("world size matches config", int(execution.get("world_size") or 0) == expected_world_size, str(execution.get("world_size")), checks)
check("tp size matches config", int(execution.get("tp_size") or 0) == expected_tp_size, str(execution.get("tp_size")), checks)
check("dtype matches config", execution.get("dtype") == expected_dtype, str(execution.get("dtype")), checks)
check("sequence length matches config", int(execution.get("sequence_length") or 0) == expected_seq_len, str(execution.get("sequence_length")), checks)
check("optimizer matches config", execution.get("optimizer_type") == expected_optimizer, str(execution.get("optimizer_type")), checks)
check("lora rank matches config", int(execution.get("lora_rank") or 0) == expected_lora_rank, str(execution.get("lora_rank")), checks)
check("measured training time is positive", as_float(measured.get("train_iteration_time_ms")) > 0, str(measured.get("train_iteration_time_ms")), checks)
check("estimated training time is positive", as_float(estimate.get("train_iteration_time_ms")) > 0, str(estimate.get("train_iteration_time_ms")), checks)
check("forward phase exists", as_float(phase_estimates.get("forward")) > 0, str(phase_estimates.get("forward")), checks)
check("backward phase exists", as_float(phase_estimates.get("backward_compute")) > 0, str(phase_estimates.get("backward_compute")), checks)
check("optimizer phase exists", as_float(phase_estimates.get("optimizer")) > 0, str(phase_estimates.get("optimizer")), checks)
if expected_parallel_mode == "tp" and expected_tp_size > 1:
    check("tp backward communication exists", as_float(phase_estimates.get("backward_comm")) > 0, str(phase_estimates.get("backward_comm")), checks)
check("dag has gradient records", int(backward.get("node_count") or 0) > 0, str(backward.get("node_count")), checks)
check("dag has layer groups", int(backward.get("layer_group_count") or 0) >= expected_layers, str(backward.get("layer_group_count")), checks)
check("dag covers every transformer layer", not missing_layers, f"missing={missing_layers}", checks)
check("dag total gradient bytes positive", int(backward.get("total_gradient_bytes") or 0) > 0, str(backward.get("total_gradient_bytes")), checks)
check("dag edges are present", len(edges) > 0, str(len(edges)), checks)
check("dag has no cycle", not graph_has_cycle(nodes, edges), "acyclic grouped graph", checks)
check(
    "run logic matches dag layers",
    int(model.get("num_hidden_layers") or 0) == expected_layers and len(layers) >= expected_layers,
    f"report_layers={model.get('num_hidden_layers')} dag_layers={len(layers)}",
    checks,
)

write_logic_dag(logic_dag_path, expected_layers, expected_parallel_mode, expected_tp_size)
check("logic dag svg written", logic_dag_path.exists(), str(logic_dag_path), checks)

ok = all(item["ok"] for item in checks)
summary_out = {
    "status": "PASS" if ok else "FAIL",
    "output_dir": str(output_dir),
    "report": str(report_path),
    "dag_dir": str(dag_dir),
    "logic_dag": str(logic_dag_path),
    "backward_layer_graph": str(dag_dir / "backward_layer_graph.svg"),
    "checks": checks,
}
(output_dir / "validation_summary.json").write_text(
    json.dumps(summary_out, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

lines = [
    "# 1-3 Train DAG Consistency Validation",
    "",
    f"- status: {summary_out['status']}",
    f"- report: {report_path}",
    f"- logic_dag: {logic_dag_path}",
    f"- backward_layer_graph: {dag_dir / 'backward_layer_graph.svg'}",
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

echo "1-3 training DAG consistency validation passed."
echo "  report: ${HOST_OUTPUT_DIR}/report.json"
echo "  validation: ${HOST_OUTPUT_DIR}/validation_report.md"
echo "  logic DAG: ${HOST_OUTPUT_DIR}/dag/logic_dag.svg"
echo "  backward DAG evidence: ${HOST_OUTPUT_DIR}/dag/backward_layer_graph.svg"
