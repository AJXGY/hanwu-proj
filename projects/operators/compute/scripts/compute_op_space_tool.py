#!/usr/bin/env python3
"""Operator-level space-dimension model for compute-intensive GEMM ops on Cambricon MLU."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from bisect import bisect_left
from collections import defaultdict

CALIBRATION_FLOPS = [
    402653184,
    805306368,
    1610612736,
    3221225472,
    6442450944,
]
VALIDATION_SEQ_LEN = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and evaluate a space model for compute-intensive ops.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--input", required=True)
    build_parser.add_argument("--model-output", required=True)
    build_parser.add_argument("--target-max-error-pct", default=20.0, type=float)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--operator", required=True)
    predict_parser.add_argument("--scale", required=True)
    predict_parser.add_argument("--flops", required=True, type=int)
    predict_parser.add_argument("--seq-len", type=int, default=None)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--model", required=True)
    eval_parser.add_argument("--input", required=True)
    eval_parser.add_argument("--summary-output", required=True)
    eval_parser.add_argument("--report-output", required=True)
    eval_parser.add_argument("--plot-dir", required=True)
    eval_parser.add_argument("--overview-plot", required=True)
    eval_parser.add_argument("--target-max-error-pct", default=20.0, type=float)
    return parser.parse_args()


def load_rows(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "operator": row["operator"],
                    "scale": row["scale"],
                    "world_size": int(row["world_size"]),
                    "seq_len": int(row["seq_len"]),
                    "m": int(row["m"]),
                    "k": int(row["k"]),
                    "n": int(row["n"]),
                    "flops": int(row["flops"]),
                    "dtype": row["dtype"],
                    "avg_ms": float(row["avg_ms"]),
                }
            )
    return rows


def grouped_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["operator"]), str(row["scale"]))].append(row)
    for value in grouped.values():
        value.sort(key=lambda item: int(item["flops"]))
    return dict(grouped)


def write_json(path: str, payload: dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_model(path: str) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_latency(model: dict[str, object], operator: str, scale: str, flops: int, seq_len: int | None = None) -> float:
    key = f"{operator}::{scale}"
    operator_model = model["operators"][key]
    points = operator_model["calibration_points"]
    axis = operator_model.get("axis", "log2_flops")
    if axis == "seq_len":
        if seq_len is None:
            raise ValueError("seq_len is required for seq_len interpolation.")
        xs = [float(point["seq_len"]) for point in points]
        x = float(seq_len)
    elif axis == "seq_len_squared":
        if seq_len is None:
            raise ValueError("seq_len is required for seq_len_squared interpolation.")
        xs = [float(point["seq_len"]) ** 2 for point in points]
        x = float(seq_len * seq_len)
    else:
        xs = [math.log2(point["flops"]) for point in points]
        x = math.log2(flops)
    ys = [point["latency_ms"] for point in points]
    if x <= xs[0]:
        left, right = 0, 1
    elif x >= xs[-1]:
        left, right = len(xs) - 2, len(xs) - 1
    else:
        right = bisect_left(xs, x)
        left = right - 1
    x0, x1 = xs[left], xs[right]
    y0, y1 = ys[left], ys[right]
    ratio = 0.0 if math.isclose(x0, x1) else (x - x0) / (x1 - x0)
    return max(y0 + ratio * (y1 - y0), 0.0)


def build_model(rows: list[dict[str, object]], target_max_error_pct: float) -> dict[str, object]:
    grouped = grouped_rows(rows)
    operators: dict[str, object] = {}
    for (operator, scale), op_rows in grouped.items():
        points_by_flops = {
            int(row["flops"]): {
                "flops": int(row["flops"]),
                "latency_ms": float(row["avg_ms"]),
                "seq_len": int(row["seq_len"]),
            }
            for row in op_rows
        }
        validation_flops = {
            int(row["flops"]) for row in op_rows if int(row["seq_len"]) == VALIDATION_SEQ_LEN
        }
        calibration_candidates = [
            int(row["flops"]) for row in op_rows if int(row["seq_len"]) != VALIDATION_SEQ_LEN
        ]
        calibration_flops = [
            value
            for value in CALIBRATION_FLOPS
            if value in points_by_flops and value not in validation_flops
        ]
        if len(calibration_flops) < 2:
            calibration_flops = sorted(calibration_candidates[:2])
        if len(calibration_flops) < 2:
            raise ValueError(f"Operator {operator} scale {scale} has too few non-{VALIDATION_SEQ_LEN} calibration points.")

        while True:
            calibration_points = [points_by_flops[value] for value in calibration_flops]
            trial_model = {
                "operators": {
                    f"{operator}::{scale}": {
                        "operator": operator,
                        "scale": scale,
                        "interpolation": "linear_axis",
                        "axis": "seq_len_squared" if operator == "flash_attention" else "seq_len",
                        "calibration_points": calibration_points,
                    }
                }
            }
            holdout_rows = [
                row
                for row in op_rows
                if int(row["seq_len"]) != VALIDATION_SEQ_LEN and int(row["flops"]) not in set(calibration_flops)
            ]
            if not holdout_rows:
                break
            worst_row = None
            worst_error = -1.0
            for row in holdout_rows:
                predicted_ms = predict_latency(trial_model, operator, scale, int(row["flops"]), int(row["seq_len"]))
                error_pct = abs(predicted_ms - float(row["avg_ms"])) / float(row["avg_ms"]) * 100.0
                if error_pct > worst_error:
                    worst_error = error_pct
                    worst_row = row
            if worst_error <= target_max_error_pct or worst_row is None:
                break
            calibration_flops.append(int(worst_row["flops"]))
            calibration_flops = sorted(set(calibration_flops))

        if len(calibration_flops) == len(calibration_candidates) and len(calibration_flops) > 2:
            removable_candidate = None
            removable_error = None
            for candidate_flops in calibration_flops:
                reduced_flops = [value for value in calibration_flops if value != candidate_flops]
                reduced_points = [points_by_flops[value] for value in reduced_flops]
                trial_model = {
                    "operators": {
                        f"{operator}::{scale}": {
                            "operator": operator,
                            "scale": scale,
                            "interpolation": "linear_axis",
                            "axis": "seq_len_squared" if operator == "flash_attention" else "seq_len",
                            "calibration_points": reduced_points,
                        }
                    }
                }
                predicted_ms = predict_latency(trial_model, operator, scale, candidate_flops, int(points_by_flops[candidate_flops]["seq_len"]))
                real_ms = float(points_by_flops[candidate_flops]["latency_ms"])
                error_pct = abs(predicted_ms - real_ms) / real_ms * 100.0
                if candidate_flops not in validation_flops and error_pct <= target_max_error_pct and (removable_error is None or error_pct < removable_error):
                    removable_candidate = candidate_flops
                    removable_error = error_pct
            if removable_candidate is not None:
                calibration_flops = [value for value in calibration_flops if value != removable_candidate]

        calibration_points = [points_by_flops[value] for value in calibration_flops]
        operators[f"{operator}::{scale}"] = {
            "operator": operator,
            "scale": scale,
            "interpolation": "linear_axis",
            "axis": "seq_len_squared" if operator == "flash_attention" else "seq_len",
            "calibration_points": calibration_points,
        }

    return {
        "tool_name": "cambricon_compute_op_space_tool",
        "tool_version": "v1",
        "device_type": "MLU580",
        "target_max_error_pct": target_max_error_pct,
        "calibration_flops": CALIBRATION_FLOPS,
        "operators": operators,
    }


def evaluate_model(model: dict[str, object], rows: list[dict[str, object]]) -> list[dict[str, object]]:
    evaluated = []
    for row in rows:
        operator = str(row["operator"])
        scale = str(row["scale"])
        predicted_ms = predict_latency(model, operator, scale, int(row["flops"]), int(row["seq_len"]))
        operator_model = model["operators"][f"{operator}::{scale}"]
        calibration_set = {int(point["flops"]) for point in operator_model["calibration_points"]}
        error_pct = abs(predicted_ms - float(row["avg_ms"])) / float(row["avg_ms"]) * 100.0
        evaluated.append(
            {
                "operator": operator,
                "scale": scale,
                "world_size": row["world_size"],
                "seq_len": row["seq_len"],
                "m": row["m"],
                "k": row["k"],
                "n": row["n"],
                "flops": row["flops"],
                "dtype": row["dtype"],
                "real_ms": row["avg_ms"],
                "sim_ms": predicted_ms,
                "error_pct": error_pct,
                "point_role": "validation" if int(row["seq_len"]) == VALIDATION_SEQ_LEN else "calibration",
            }
        )
    return evaluated


def write_evaluation_csv(path: str, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "operator",
        "scale",
        "world_size",
        "seq_len",
        "m",
        "k",
        "n",
        "flops",
        "dtype",
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
                    **row,
                    "real_ms": f"{row['real_ms']:.6f}",
                    "sim_ms": f"{row['sim_ms']:.6f}",
                    "error_pct": f"{row['error_pct']:.4f}",
                }
            )


def write_report_csv(path: str, rows: list[dict[str, object]], model: dict[str, object], target_max_error_pct: float) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["operator"]), str(row["scale"]))].append(row)
    fieldnames = [
        "operator",
        "scale",
        "calibration_points",
        "validation_points",
        "validation_avg_error_pct",
        "validation_max_error_pct",
        "target_max_error_pct",
        "pass_le_target_pct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (operator, scale), op_rows in sorted(grouped.items()):
            validation_rows = [row for row in op_rows if row["point_role"] == "validation"]
            validation_errors = [float(row["error_pct"]) for row in validation_rows]
            avg_validation_error = sum(validation_errors) / len(validation_errors) if validation_errors else 0.0
            max_validation_error = max(validation_errors) if validation_errors else 0.0
            writer.writerow(
                {
                    "operator": operator,
                    "scale": scale,
                    "calibration_points": len(model["operators"][f"{operator}::{scale}"]["calibration_points"]),
                    "validation_points": len(validation_rows),
                    "validation_avg_error_pct": f"{avg_validation_error:.4f}",
                    "validation_max_error_pct": f"{max_validation_error:.4f}",
                    "target_max_error_pct": f"{target_max_error_pct:.4f}",
                    "pass_le_target_pct": "yes" if max_validation_error <= target_max_error_pct else "no",
                }
            )


def make_plots(plot_dir: str, overview_path: str, rows: list[dict[str, object]], target_max_error_pct: float) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["operator"]), str(row["scale"]))].append(row)

    overview_labels = []
    overview_errors = []

    for (operator, scale), op_rows in sorted(grouped.items()):
        op_rows.sort(key=lambda item: int(item["flops"]))
        xs = [int(row["flops"]) for row in op_rows]
        real = [float(row["real_ms"]) for row in op_rows]
        sim = [float(row["sim_ms"]) for row in op_rows]
        errors = [float(row["error_pct"]) for row in op_rows]
        roles = [str(row["point_role"]) for row in op_rows]
        validation_errors = [err for err, role in zip(errors, roles) if role == "validation"]
        overview_labels.append(f"{operator}\n{scale}")
        overview_errors.append(max(validation_errors) if validation_errors else 0.0)

        fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), dpi=180, gridspec_kw={"height_ratios": [3, 2]})
        ax_top, ax_bottom = axes
        ax_top.plot(xs, real, marker="o", linewidth=2.4, color="#0f4c5c", label="Measured")
        ax_top.plot(xs, sim, marker="s", linewidth=2.0, linestyle="--", color="#d97706", label="Predicted")
        calibration_xs = [x for x, role in zip(xs, roles) if role == "calibration"]
        calibration_ys = [y for y, role in zip(real, roles) if role == "calibration"]
        ax_top.scatter(calibration_xs, calibration_ys, s=46, color="#2f6f4f", edgecolors="white", linewidths=0.7, zorder=5)
        ax_top.set_xscale("log", base=2)
        ax_top.set_ylabel("Latency (ms)")
        ax_top.grid(True, alpha=0.2)
        ax_top.legend(frameon=False, loc="upper left")
        ax_top.set_title(f"{operator} / {scale}")

        top_error = max(max(errors) * 1.15, target_max_error_pct + 2.0)
        ax_bottom.axhspan(0.0, target_max_error_pct, color="#dfeee2", alpha=0.8)
        ax_bottom.axhspan(target_max_error_pct, top_error, color="#f6ddd4", alpha=0.6)
        ax_bottom.bar(xs, errors, width=[x * 0.15 for x in xs], color="#d66b4d", alpha=0.72)
        ax_bottom.axhline(target_max_error_pct, color="#2f6f4f", linestyle="--", linewidth=1.6)
        ax_bottom.set_xscale("log", base=2)
        ax_bottom.set_xlabel("FLOPs")
        ax_bottom.set_ylabel("Error (%)")
        ax_bottom.grid(True, alpha=0.2)
        ax_bottom.set_ylim(0, max(target_max_error_pct + 2.0, max(errors) * 1.18))

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{operator}_{scale}_strict_validation.png"), bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.4, 5.4), dpi=180)
    bars = ax.bar(range(len(overview_labels)), overview_errors, color="#3a7ca5", alpha=0.82)
    ax.axhline(target_max_error_pct, color="#c13c37", linestyle="--", linewidth=1.8, label=f"{target_max_error_pct:.0f}% limit")
    ax.set_xticks(range(len(overview_labels)))
    ax.set_xticklabels(overview_labels, rotation=25, ha="right")
    ax.set_ylabel("Max Validation Error (%)")
    ax.set_title("Compute-Intensive Operator Space-Model Validation")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    for bar, value in zip(bars, overview_errors):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(overview_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.command == "build":
        model = build_model(load_rows(args.input), args.target_max_error_pct)
        write_json(args.model_output, model)
        return
    if args.command == "predict":
        model = load_model(args.model)
        print(f"{predict_latency(model, args.operator, args.scale, args.flops, args.seq_len):.6f}")
        return
    if args.command == "evaluate":
        model = load_model(args.model)
        rows = load_rows(args.input)
        evaluated = evaluate_model(model, rows)
        write_evaluation_csv(args.summary_output, evaluated)
        write_report_csv(args.report_output, evaluated, model, args.target_max_error_pct)
        make_plots(args.plot_dir, args.overview_plot, evaluated, args.target_max_error_pct)
        return


if __name__ == "__main__":
    main()
