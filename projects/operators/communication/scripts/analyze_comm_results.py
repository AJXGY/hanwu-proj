#!/usr/bin/env python3
"""Fit a simple communication model and generate summary tables and plots.

Input: benchmark CSV with real latency measurements.
Output: per-point model/error summary and a comparison figure.
"""

import argparse
import csv
import math
import os
from typing import Dict, List

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Cambricon communication benchmark results."
    )
    parser.add_argument(
        "--input",
        default="results/processed/comm_bench_combined.csv",
        help="Benchmark CSV file.",
    )
    parser.add_argument(
        "--summary-output",
        default="results/processed/comm_model_summary.csv",
        help="Per-point summary CSV file.",
    )
    parser.add_argument(
        "--plot-output",
        default="figure/comm_model_vs_real.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--per-operator-dir",
        default="figure/operators",
        help="Directory for per-operator plots.",
    )
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
                    "min_ms": float(row["min_ms"]),
                    "max_ms": float(row["max_ms"]),
                    "std_ms": float(row["std_ms"]),
                }
            )
    return rows


def fit_linear_model(xs: List[float], ys: List[float]) -> Dict[str, float]:
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if math.isclose(denom, 0.0):
        beta = 0.0
    else:
        beta = (n * sum_xy - sum_x * sum_y) / denom
    alpha = (sum_y - beta * sum_x) / n
    return {"alpha": alpha, "beta": beta}


def predict_segmented(message_bytes: int, model: Dict[str, object]) -> float:
    predicted_ms = 0.0
    for segment in model["segments"]:
        if message_bytes <= segment["upper_bytes"]:
            predicted_ms = segment["alpha"] + segment["beta"] * message_bytes
            break
    else:
        last_segment = model["segments"][-1]
        predicted_ms = last_segment["alpha"] + last_segment["beta"] * message_bytes
    return max(predicted_ms, 0.0)


def fit_best_segmented_model(
    op_rows: List[Dict[str, float]],
    max_segments: int = 4,
    min_points_per_segment: int = 2,
) -> Dict[str, object]:
    op_rows = sorted(op_rows, key=lambda item: item["message_bytes"])
    points = [(row["message_bytes"], row["avg_ms"]) for row in op_rows]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    point_count = len(points)
    max_segments = max(1, min(max_segments, point_count // min_points_per_segment))

    segment_models: List[List[Dict[str, float]]] = [[{} for _ in range(point_count)] for _ in range(point_count)]
    segment_errors: List[List[float]] = [[0.0 for _ in range(point_count)] for _ in range(point_count)]

    for start in range(point_count):
        for end in range(start, point_count):
            model = fit_linear_model(xs[start : end + 1], ys[start : end + 1])
            errors = [
                abs(max(model["alpha"] + model["beta"] * xs[idx], 0.0) - ys[idx]) / ys[idx] * 100.0
                for idx in range(start, end + 1)
            ]
            segment_models[start][end] = model
            segment_errors[start][end] = max(errors)

    inf = float("inf")
    dp: List[List[float]] = [[inf for _ in range(point_count)] for _ in range(max_segments + 1)]
    cut: List[List[int]] = [[-1 for _ in range(point_count)] for _ in range(max_segments + 1)]

    for end in range(min_points_per_segment - 1, point_count):
        dp[1][end] = segment_errors[0][end]

    for segment_count in range(2, max_segments + 1):
        min_end = segment_count * min_points_per_segment - 1
        min_prev_end = (segment_count - 1) * min_points_per_segment - 1
        for end in range(min_end, point_count):
            max_prev_end = end - min_points_per_segment
            for prev_end in range(min_prev_end, max_prev_end + 1):
                score = max(dp[segment_count - 1][prev_end], segment_errors[prev_end + 1][end])
                if score < dp[segment_count][end]:
                    dp[segment_count][end] = score
                    cut[segment_count][end] = prev_end

    best_segment_count = min(range(1, max_segments + 1), key=lambda count: dp[count][point_count - 1])

    reconstructed = []
    current_count = best_segment_count
    current_end = point_count - 1
    while current_count >= 1:
        previous_end = cut[current_count][current_end] if current_count > 1 else -1
        current_start = previous_end + 1
        current_model = segment_models[current_start][current_end]
        reconstructed.append(
            {
                "lower_bytes": xs[current_start],
                "upper_bytes": xs[current_end],
                "alpha": current_model["alpha"],
                "beta": current_model["beta"],
            }
        )
        current_end = previous_end
        current_count -= 1
    reconstructed.reverse()

    return {
        "model_type": "segmented_linear",
        "segment_count": best_segment_count,
        "thresholds": [segment["upper_bytes"] for segment in reconstructed[:-1]],
        "segments": reconstructed,
    }


def analyze(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(row["operator"], []).append(row)

    analyzed_rows: List[Dict[str, float]] = []
    for operator, op_rows in grouped.items():
        op_rows.sort(key=lambda item: item["message_bytes"])
        model = fit_best_segmented_model(op_rows)
        for row in op_rows:
            predicted_ms = predict_segmented(row["message_bytes"], model)
            error_pct = abs(predicted_ms - row["avg_ms"]) / row["avg_ms"] * 100.0
            analyzed_rows.append(
                {
                    "operator": operator,
                    "message_bytes": row["message_bytes"],
                    "real_ms": row["avg_ms"],
                    "sim_ms": predicted_ms,
                    "error_pct": error_pct,
                    "model_type": model["model_type"],
                    "segment_count": model["segment_count"],
                    "thresholds": model["thresholds"],
                    "segment_ranges": [
                        f"{segment['lower_bytes']}-{segment['upper_bytes']}"
                        for segment in model["segments"]
                    ],
                    "segment_models": [
                        f"{segment['alpha']:.6f},{segment['beta']:.12f}"
                        for segment in model["segments"]
                    ],
                }
            )
    return analyzed_rows


def write_summary(path: str, rows: List[Dict[str, float]]) -> None:
    fieldnames = [
        "operator",
        "message_bytes",
        "real_ms",
        "sim_ms",
        "error_pct",
        "model_type",
        "segment_count",
        "threshold_bytes",
        "segment_ranges_bytes",
        "segment_models",
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
                    "model_type": row["model_type"],
                    "segment_count": row["segment_count"],
                    "threshold_bytes": ";".join(str(value) for value in row["thresholds"]),
                    "segment_ranges_bytes": ";".join(row["segment_ranges"]),
                    "segment_models": ";".join(row["segment_models"]),
                }
            )


def make_plot(path: str, rows: List[Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(row["operator"], []).append(row)

    operators = sorted(grouped.items())
    max_cols = min(3, len(operators))
    group_rows = math.ceil(len(operators) / max_cols)
    fig, axes = plt.subplots(
        2 * group_rows,
        max_cols,
        figsize=(7.2 * max_cols, 4.0 * group_rows + 1.8),
        dpi=200,
        gridspec_kw={"height_ratios": [value for _ in range(group_rows) for value in (3, 2)]},
    )
    total_axes_rows = 2 * group_rows
    if max_cols == 1:
        axes = [[row] for row in axes]
    else:
        axes = [list(row) for row in axes]

    fig.patch.set_facecolor("#f6f1e8")
    real_color = "#0f4c5c"
    sim_color = "#d97706"
    error_color = "#a62c1e"
    threshold_color = "#2f6f4f"
    pass_color = "#dfeee2"
    fail_color = "#f6ddd4"

    def format_bytes(value: float, _pos: float) -> str:
        if value >= 1024**2:
            return f"{value / (1024**2):.0f} MiB"
        if value >= 1024:
            return f"{value / 1024:.0f} KiB"
        return f"{value:.0f} B"

    for idx, (operator, op_rows) in enumerate(operators):
        group_row = idx // max_cols
        col = idx % max_cols
        ax_top = axes[group_row * 2][col]
        ax_bottom = axes[group_row * 2 + 1][col]
        op_rows.sort(key=lambda item: item["message_bytes"])
        xs = [row["message_bytes"] for row in op_rows]
        real = [row["real_ms"] for row in op_rows]
        sim = [row["sim_ms"] for row in op_rows]
        errors = [row["error_pct"] for row in op_rows]
        thresholds = op_rows[0]["thresholds"]
        threshold_label = ", ".join(f"{value / 1024:.0f} KiB" for value in thresholds) if thresholds else "none"
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        status = "PASS" if max_error <= 20.0 else "FAIL"

        ax_top.set_facecolor("#fffdfa")
        ax_top.plot(
            xs,
            real,
            marker="o",
            markersize=5.4,
            linewidth=2.8,
            color=real_color,
            label="Measured latency",
        )
        ax_top.plot(
            xs,
            sim,
            marker="s",
            markersize=5.0,
            linewidth=2.2,
            linestyle="--",
            color=sim_color,
            label="Predicted latency",
        )
        for threshold_idx, threshold in enumerate(thresholds):
            ax_top.axvline(
                threshold,
                color=threshold_color,
                linestyle=":",
                linewidth=1.8,
                label="Breakpoints" if threshold_idx == 0 else None,
            )
        ax_top.set_xscale("log", base=2)
        ax_top.set_ylabel("Latency (ms)")
        ax_top.set_title(operator.replace("_", " ").title(), fontsize=13, weight="bold")
        ax_top.grid(True, alpha=0.18, linewidth=0.8)
        ax_top.xaxis.set_major_formatter(FuncFormatter(format_bytes))
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.legend(frameon=False, loc="upper left")
        summary_text = (
            f"{status}\n"
            f"avg error: {avg_error:.2f}%\n"
            f"max error: {max_error:.2f}%\n"
            f"segments: {op_rows[0]['segment_count']}\n"
            f"breakpoints: {threshold_label}"
        )
        ax_top.text(
            0.98,
            0.05,
            summary_text,
            transform=ax_top.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.3,
            color="#173f35",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "#f0efe7",
                "edgecolor": "#d7cfbf",
            },
        )

        ax_bottom.set_facecolor("#fffdfa")
        ax_bottom.axhspan(0.0, 20.0, color=pass_color, alpha=0.85, zorder=0)
        ax_bottom.axhspan(20.0, max(22.0, max_error * 1.15), color=fail_color, alpha=0.6, zorder=0)
        ax_bottom.fill_between(xs, errors, [0.0] * len(errors), color="#edc9bb", alpha=0.72)
        ax_bottom.plot(
            xs,
            errors,
            marker="D",
            markersize=4.5,
            linewidth=1.9,
            color=error_color,
            label="Absolute error",
        )
        ax_bottom.axhline(
            20.0,
            color=threshold_color,
            linestyle="--",
            linewidth=1.6,
            label="Acceptance limit (20%)",
        )
        for threshold in thresholds:
            ax_bottom.axvline(
                threshold,
                color=threshold_color,
                linestyle=":",
                linewidth=1.8,
            )
        ax_bottom.set_xscale("log", base=2)
        ax_bottom.set_xlabel("Message Size (Bytes)")
        ax_bottom.set_ylabel("Error (%)")
        ax_bottom.grid(True, alpha=0.18, linewidth=0.8)
        ax_bottom.xaxis.set_major_formatter(FuncFormatter(format_bytes))
        ax_bottom.set_ylim(0, max(22.0, max_error * 1.18))
        ax_bottom.legend(frameon=False, loc="upper right")

    for row_idx in range(total_axes_rows):
        for col_idx in range(max_cols):
            operator_idx = (row_idx // 2) * max_cols + col_idx
            if operator_idx >= len(operators):
                axes[row_idx][col_idx].axis("off")

    fig.suptitle(
        "Cambricon MLU Communication Operator Validation\nMeasured Latency vs Segmented Model",
        fontsize=16.5,
        weight="semibold",
        fontfamily="DejaVu Serif",
        y=0.985,
    )
    fig.text(
        0.5,
        0.915,
        "Criterion: every measured point must stay within 20% prediction error on single-node dual-MLU runs",
        ha="center",
        va="center",
        fontsize=10.0,
        color="#5b5b5b",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(path, bbox_inches="tight")


def make_per_operator_plots(output_dir: str, rows: List[Dict[str, float]]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(row["operator"], []).append(row)

    for operator, op_rows in grouped.items():
        figure_path = os.path.join(output_dir, f"{operator}_model_vs_real.png")
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(8.8, 7.2),
            dpi=200,
            gridspec_kw={"height_ratios": [3, 2]},
        )
        fig.patch.set_facecolor("#f6f1e8")
        ax_top, ax_bottom = axes

        real_color = "#0f4c5c"
        sim_color = "#d97706"
        error_color = "#a62c1e"
        threshold_color = "#2f6f4f"
        pass_color = "#dfeee2"
        fail_color = "#f6ddd4"

        def format_bytes(value: float, _pos: float) -> str:
            if value >= 1024**2:
                return f"{value / (1024**2):.0f} MiB"
            if value >= 1024:
                return f"{value / 1024:.0f} KiB"
            return f"{value:.0f} B"

        ordered_rows = sorted(op_rows, key=lambda item: item["message_bytes"])
        xs = [row["message_bytes"] for row in ordered_rows]
        real = [row["real_ms"] for row in ordered_rows]
        sim = [row["sim_ms"] for row in ordered_rows]
        errors = [row["error_pct"] for row in ordered_rows]
        thresholds = ordered_rows[0]["thresholds"]
        threshold_label = ", ".join(f"{value / 1024:.0f} KiB" for value in thresholds) if thresholds else "none"
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        status = "PASS" if max_error <= 20.0 else "FAIL"

        ax_top.set_facecolor("#fffdfa")
        ax_top.plot(xs, real, marker="o", markersize=5.4, linewidth=2.8, color=real_color, label="Measured latency")
        ax_top.plot(xs, sim, marker="s", markersize=5.0, linewidth=2.2, linestyle="--", color=sim_color, label="Predicted latency")
        for threshold_idx, threshold in enumerate(thresholds):
            ax_top.axvline(
                threshold,
                color=threshold_color,
                linestyle=":",
                linewidth=1.8,
                label="Breakpoints" if threshold_idx == 0 else None,
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
                f"{status}\n"
                f"avg error: {avg_error:.2f}%\n"
                f"max error: {max_error:.2f}%\n"
                f"segments: {ordered_rows[0]['segment_count']}\n"
                f"breakpoints: {threshold_label}"
            ),
            transform=ax_top.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.6,
            color="#173f35",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "#f0efe7",
                "edgecolor": "#d7cfbf",
            },
        )

        ax_bottom.set_facecolor("#fffdfa")
        ax_bottom.axhspan(0.0, 20.0, color=pass_color, alpha=0.85, zorder=0)
        ax_bottom.axhspan(20.0, max(22.0, max_error * 1.15), color=fail_color, alpha=0.6, zorder=0)
        ax_bottom.fill_between(xs, errors, [0.0] * len(errors), color="#edc9bb", alpha=0.72)
        ax_bottom.plot(xs, errors, marker="D", markersize=4.5, linewidth=1.9, color=error_color, label="Absolute error")
        ax_bottom.axhline(20.0, color=threshold_color, linestyle="--", linewidth=1.6, label="Acceptance limit (20%)")
        for threshold in thresholds:
            ax_bottom.axvline(threshold, color=threshold_color, linestyle=":", linewidth=1.8)
        ax_bottom.set_xscale("log", base=2)
        ax_bottom.set_xlabel("Message Size (Bytes)")
        ax_bottom.set_ylabel("Error (%)")
        ax_bottom.grid(True, alpha=0.18, linewidth=0.8)
        ax_bottom.xaxis.set_major_formatter(FuncFormatter(format_bytes))
        ax_bottom.set_ylim(0, max(22.0, max_error * 1.18))
        ax_bottom.legend(frameon=False, loc="upper right")

        fig.suptitle(
            f"{operator.replace('_', ' ').title()} Communication Model Validation",
            fontsize=17,
            weight="semibold",
            fontfamily="DejaVu Serif",
            y=0.975,
        )
        fig.text(
            0.5,
            0.93,
            "Single-node dual-MLU measured latency vs segmented model",
            ha="center",
            va="center",
            fontsize=10.1,
            color="#5b5b5b",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.86])
        fig.savefig(figure_path, bbox_inches="tight")
        plt.close(fig)


def print_report(rows: List[Dict[str, float]]) -> None:
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(row["operator"], []).append(row)
    for operator, op_rows in grouped.items():
        max_error = max(row["error_pct"] for row in op_rows)
        avg_error = sum(row["error_pct"] for row in op_rows) / len(op_rows)
        thresholds = ",".join(str(value) for value in op_rows[0]["thresholds"]) or "none"
        print(
            f"{operator}: model=segmented_linear({op_rows[0]['segment_count']}), "
            f"thresholds={thresholds} bytes, "
            f"avg_error={avg_error:.2f}%, max_error={max_error:.2f}%"
        )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    analyzed_rows = analyze(rows)
    write_summary(args.summary_output, analyzed_rows)
    make_plot(args.plot_output, analyzed_rows)
    make_per_operator_plots(args.per_operator_dir, analyzed_rows)
    print_report(analyzed_rows)


if __name__ == "__main__":
    main()
