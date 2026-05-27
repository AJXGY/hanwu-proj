#!/usr/bin/env python3
"""Operator-level response-time analysis tool for Cambricon communication ops.

This tool builds a sparse calibration model from fixed message sizes and then
predicts latency for arbitrary message sizes using byte-axis interpolation.
It is intended to serve as a standalone, callable response-time analysis tool
for the communication-operator validation task.
"""

import argparse
import csv
import json
import math
import os
from bisect import bisect_left
from collections import defaultdict
from typing import Dict, List

CALIBRATION_MESSAGE_BYTES = [
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    12582912,
    16777216,
    29360128,
    54525952,
    134217728,
]
SEQ1024_VALIDATION_MESSAGE_BYTES = 8388608


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and evaluate an operator-level response-time analysis tool."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a sparse calibration model.")
    build_parser.add_argument("--input", required=True, help="Benchmark CSV path.")
    build_parser.add_argument("--model-output", required=True, help="Model JSON output path.")

    predict_parser = subparsers.add_parser("predict", help="Predict one operator latency.")
    predict_parser.add_argument("--model", required=True, help="Model JSON path.")
    predict_parser.add_argument("--operator", required=True, help="Operator name.")
    predict_parser.add_argument("--message-bytes", required=True, type=int, help="Message size in bytes.")
    predict_parser.add_argument("--world-size", default=2, type=int, help="Parallel world size.")
    predict_parser.add_argument("--device-type", default="MLU580", help="Device type.")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the tool on benchmark CSV.")
    eval_parser.add_argument("--model", required=True, help="Model JSON path.")
    eval_parser.add_argument("--input", required=True, help="Benchmark CSV path.")
    eval_parser.add_argument("--summary-output", required=True, help="Per-point evaluation CSV.")
    eval_parser.add_argument("--report-output", required=True, help="Per-operator summary CSV.")
    eval_parser.add_argument("--plot-dir", required=True, help="Per-operator figure directory.")

    return parser.parse_args()


def load_rows(path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "operator": row["operator"],
                    "message_bytes": int(row["message_bytes"]),
                    "avg_ms": float(row["avg_ms"]),
                }
            )
    return rows


def group_rows(rows: List[Dict[str, float]]) -> Dict[str, List[Dict[str, float]]]:
    grouped: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["operator"]].append(row)
    for operator_rows in grouped.values():
        operator_rows.sort(key=lambda item: item["message_bytes"])
    return dict(grouped)


def build_model(rows: List[Dict[str, float]]) -> Dict[str, object]:
    if SEQ1024_VALIDATION_MESSAGE_BYTES in CALIBRATION_MESSAGE_BYTES:
        raise ValueError("seq=1024 validation point must not be used as a calibration point.")

    grouped = group_rows(rows)
    operators: Dict[str, object] = {}

    for operator, operator_rows in grouped.items():
        calibration_points = [
            {"message_bytes": row["message_bytes"], "latency_ms": row["avg_ms"]}
            for row in operator_rows
            if row["message_bytes"] in CALIBRATION_MESSAGE_BYTES
        ]
        if len(calibration_points) != len(CALIBRATION_MESSAGE_BYTES):
            raise ValueError(
                f"Operator {operator} missing calibration points. "
                f"Expected {len(CALIBRATION_MESSAGE_BYTES)}, got {len(calibration_points)}."
            )
        operators[operator] = {
            "interpolation": "linear_axis",
            "axis": "message_bytes",
            "calibration_points": calibration_points,
        }

    return {
        "tool_name": "cambricon_comm_response_time_tool",
        "tool_version": "v1",
        "device_type": "MLU580",
        "world_size": 2,
        "calibration_message_bytes": CALIBRATION_MESSAGE_BYTES,
        "operators": operators,
    }


def write_model(path: str, model: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)


def load_model(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_latency(model: Dict[str, object], operator: str, message_bytes: int) -> float:
    operator_model = model["operators"][operator]
    points = operator_model["calibration_points"]
    axis = operator_model.get("axis", "message_bytes")
    if axis == "log2_message_bytes":
        xs = [math.log2(point["message_bytes"]) for point in points]
        message_axis = math.log2(message_bytes)
    else:
        xs = [float(point["message_bytes"]) for point in points]
        message_axis = float(message_bytes)
    ys = [point["latency_ms"] for point in points]

    if message_axis <= xs[0]:
        left_idx, right_idx = 0, 1
    elif message_axis >= xs[-1]:
        left_idx, right_idx = len(xs) - 2, len(xs) - 1
    else:
        right_idx = bisect_left(xs, message_axis)
        left_idx = right_idx - 1

    x0, x1 = xs[left_idx], xs[right_idx]
    y0, y1 = ys[left_idx], ys[right_idx]
    ratio = 0.0 if math.isclose(x0, x1) else (message_axis - x0) / (x1 - x0)
    predicted_ms = y0 + ratio * (y1 - y0)
    return max(predicted_ms, 0.0)


def evaluate_model(model: Dict[str, object], rows: List[Dict[str, float]]) -> List[Dict[str, object]]:
    evaluated_rows: List[Dict[str, object]] = []
    calibration_set = set(model["calibration_message_bytes"])
    for row in rows:
        predicted_ms = predict_latency(model, row["operator"], row["message_bytes"])
        error_pct = abs(predicted_ms - row["avg_ms"]) / row["avg_ms"] * 100.0
        evaluated_rows.append(
            {
                "operator": row["operator"],
                "message_bytes": row["message_bytes"],
                "real_ms": row["avg_ms"],
                "sim_ms": predicted_ms,
                "error_pct": error_pct,
                "point_role": "calibration" if row["message_bytes"] in calibration_set else "validation",
            }
        )
    return evaluated_rows


def write_evaluation_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "operator",
        "message_bytes",
        "real_ms",
        "sim_ms",
        "error_pct",
        "point_role",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "operator": row["operator"],
                    "message_bytes": row["message_bytes"],
                    "real_ms": f"{row['real_ms']:.6f}",
                    "sim_ms": f"{row['sim_ms']:.6f}",
                    "error_pct": f"{row['error_pct']:.4f}",
                    "point_role": row["point_role"],
                }
            )


def write_report_csv(path: str, rows: List[Dict[str, object]], model: Dict[str, object]) -> None:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[row["operator"]].append(row)

    fieldnames = [
        "operator",
        "calibration_points",
        "validation_points",
        "validation_avg_error_pct",
        "validation_max_error_pct",
        "all_points_avg_error_pct",
        "all_points_max_error_pct",
        "pass_le_20_pct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for operator, operator_rows in sorted(grouped.items()):
            validation_rows = [row for row in operator_rows if row["point_role"] == "validation"]
            all_errors = [row["error_pct"] for row in operator_rows]
            validation_errors = [row["error_pct"] for row in validation_rows]
            writer.writerow(
                {
                    "operator": operator,
                    "calibration_points": len(model["operators"][operator]["calibration_points"]),
                    "validation_points": len(validation_rows),
                    "validation_avg_error_pct": f"{sum(validation_errors) / len(validation_errors):.4f}",
                    "validation_max_error_pct": f"{max(validation_errors):.4f}",
                    "all_points_avg_error_pct": f"{sum(all_errors) / len(all_errors):.4f}",
                    "all_points_max_error_pct": f"{max(all_errors):.4f}",
                    "pass_le_20_pct": "yes" if max(validation_errors) <= 20.0 else "no",
                }
            )


def format_bytes(value: float, _pos: float) -> str:
    if value >= 1024**2:
        return f"{value / (1024**2):.0f} MiB"
    if value >= 1024:
        return f"{value / 1024:.0f} KiB"
    return f"{value:.0f} B"


def make_plots(plot_dir: str, rows: List[Dict[str, object]], model: Dict[str, object]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "legend.fontsize": 10,
        }
    )

    os.makedirs(plot_dir, exist_ok=True)
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[row["operator"]].append(row)

    for operator, operator_rows in sorted(grouped.items()):
        operator_rows.sort(key=lambda item: item["message_bytes"])
        xs = [row["message_bytes"] for row in operator_rows]
        real = [row["real_ms"] for row in operator_rows]
        sim = [row["sim_ms"] for row in operator_rows]
        errors = [row["error_pct"] for row in operator_rows]
        roles = [row["point_role"] for row in operator_rows]
        validation_errors = [row["error_pct"] for row in operator_rows if row["point_role"] == "validation"]
        avg_validation_error = sum(validation_errors) / len(validation_errors)
        max_validation_error = max(validation_errors)

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(8.8, 7.2),
            dpi=200,
            gridspec_kw={"height_ratios": [3, 2]},
        )
        fig.patch.set_facecolor("#f6f1e8")
        ax_top, ax_bottom = axes

        ax_top.set_facecolor("#fffdfa")
        ax_top.plot(xs, real, marker="o", markersize=5.2, linewidth=2.6, color="#0f4c5c", label="Measured latency")
        ax_top.plot(xs, sim, marker="s", markersize=4.8, linewidth=2.2, linestyle="--", color="#d97706", label="Tool prediction")
        calibration_xs = [x for x, role in zip(xs, roles) if role == "calibration"]
        calibration_ys = [y for y, role in zip(real, roles) if role == "calibration"]
        ax_top.scatter(
            calibration_xs,
            calibration_ys,
            s=45,
            color="#2f6f4f",
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
            label="Calibration points",
        )
        ax_top.set_xscale("log", base=2)
        ax_top.set_ylabel("Latency (ms)")
        ax_top.grid(True, alpha=0.18, linewidth=0.8)
        ax_top.xaxis.set_major_formatter(FuncFormatter(format_bytes))
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.legend(frameon=False, loc="upper left")
        ax_top.text(
            0.98,
            0.05,
            (
                f"validation avg: {avg_validation_error:.2f}%\n"
                f"validation max: {max_validation_error:.2f}%\n"
                f"calibration points: {len(calibration_xs)}\n"
                f"validation points: {len(validation_errors)}"
            ),
            transform=ax_top.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.4,
            color="#173f35",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "#f0efe7",
                "edgecolor": "#d7cfbf",
            },
        )

        ax_bottom.set_facecolor("#fffdfa")
        ax_bottom.axhspan(0.0, 20.0, color="#dfeee2", alpha=0.85, zorder=0)
        ax_bottom.axhspan(20.0, max(22.0, max(errors) * 1.15), color="#f6ddd4", alpha=0.6, zorder=0)
        ax_bottom.fill_between(xs, errors, [0.0] * len(errors), color="#edc9bb", alpha=0.72)
        ax_bottom.plot(xs, errors, marker="D", markersize=4.3, linewidth=1.8, color="#a62c1e", label="Absolute error")
        ax_bottom.axhline(20.0, color="#2f6f4f", linestyle="--", linewidth=1.6, label="Acceptance limit (20%)")
        ax_bottom.set_xscale("log", base=2)
        ax_bottom.set_xlabel("Message Size (Bytes)")
        ax_bottom.set_ylabel("Error (%)")
        ax_bottom.grid(True, alpha=0.18, linewidth=0.8)
        ax_bottom.xaxis.set_major_formatter(FuncFormatter(format_bytes))
        ax_bottom.set_ylim(0, max(22.0, max(errors) * 1.18))
        ax_bottom.legend(frameon=False, loc="upper right")

        fig.suptitle(
            f"{operator.replace('_', ' ').title()} Response-Time Tool Validation",
            fontsize=17,
            weight="semibold",
            fontfamily="DejaVu Serif",
            y=0.975,
        )
        fig.text(
            0.5,
            0.93,
            "Sparse calibration profile + interpolation on single-node dual-MLU communication measurements",
            ha="center",
            va="center",
            fontsize=10.0,
            color="#5b5b5b",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.86])
        fig.savefig(os.path.join(plot_dir, f"{operator}_strict_validation.png"), bbox_inches="tight")
        plt.close(fig)


def print_report(rows: List[Dict[str, object]]) -> None:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[row["operator"]].append(row)

    for operator, operator_rows in sorted(grouped.items()):
        validation_rows = [row for row in operator_rows if row["point_role"] == "validation"]
        validation_errors = [row["error_pct"] for row in validation_rows]
        print(
            f"{operator}: validation_points={len(validation_rows)}, "
            f"avg_error={sum(validation_errors) / len(validation_errors):.2f}%, "
            f"max_error={max(validation_errors):.2f}%"
        )


def main() -> None:
    args = parse_args()
    if args.command == "build":
        rows = load_rows(args.input)
        model = build_model(rows)
        write_model(args.model_output, model)
        print(f"Built model into {args.model_output}")
        return

    if args.command == "predict":
        model = load_model(args.model)
        if args.world_size != model["world_size"]:
            raise ValueError(f"Model world_size={model['world_size']}, got {args.world_size}")
        if args.device_type != model["device_type"]:
            raise ValueError(f"Model device_type={model['device_type']}, got {args.device_type}")
        predicted_ms = predict_latency(model, args.operator, args.message_bytes)
        print(f"{predicted_ms:.6f}")
        return

    if args.command == "evaluate":
        model = load_model(args.model)
        rows = load_rows(args.input)
        evaluated_rows = evaluate_model(model, rows)
        write_evaluation_csv(args.summary_output, evaluated_rows)
        write_report_csv(args.report_output, evaluated_rows, model)
        make_plots(args.plot_dir, evaluated_rows, model)
        print_report(evaluated_rows)
        return


if __name__ == "__main__":
    main()
