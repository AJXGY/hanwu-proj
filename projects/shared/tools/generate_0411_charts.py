from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_DIR = REPO_ROOT / "docs" / "history" / "2026-04-11-training-inference" / "charts" / "0411"

PALETTE = {
    "bg": (247, 242, 231),
    "panel": (255, 250, 241),
    "line": (213, 200, 179),
    "grid": (231, 221, 204),
    "text": (30, 26, 22),
    "muted": (109, 100, 87),
    "green": (29, 107, 82),
    "orange": (201, 109, 45),
    "red": (178, 74, 44),
    "brown": (139, 61, 33),
    "white": (255, 255, 255),
}

FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "%": ["11001", "11010", "00100", "01000", "10110", "00110", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    "(": ["00110", "01100", "01100", "01100", "01100", "01100", "00110"],
    ")": ["01100", "00110", "00110", "00110", "00110", "00110", "01100"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00001", "00001", "00001", "00001", "10001", "10001", "01110"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "10001", "11001", "10101", "10011", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inference_rows() -> list[dict]:
    tp = load_json(
        "/home/o_mabin/hanwu-proj/projects/inference/time-modeling/validation_reports/cambricon_tp2_smoke/report.json"
    )
    rows = [
        {
            "name": "TP2",
            "error": float(tp["comparison"]["request_relative_error_pct"]),
            "measured": float(tp["measured"]["request"]["mean_ms"]),
            "estimated": float(tp["estimate"]["request_end_to_end_time_ms"]),
        }
    ]
    for name in ["pp1_mb2", "pp2_mb1", "pp2_mb2", "pp2_mb4"]:
        report = load_json(
            f"/home/o_mabin/hanwu-proj/projects/inference/time-modeling/validation_reports/cambricon_pp_smoke/{name}/report.json"
        )
        rows.append(
            {
                "name": name.upper(),
                "error": float(report["comparison"]["request_relative_error_pct"]),
                "measured": float(report["measured"]["request"]["mean_ms"]),
                "estimated": float(report["estimate"]["request_makespan_ms"]),
            }
        )
    return rows


def training_rows() -> list[dict]:
    rows = []
    for group, names in [
        ("real8b_pp1_training", ["pp1_mb1", "pp1_mb2", "pp1_mb4"]),
        ("cambricon_train_smoke", ["pp2_mb1", "pp2_mb2", "pp2_mb4"]),
    ]:
        for name in names:
            report = load_json(
                f"/home/o_mabin/hanwu-proj/projects/training/time-modeling/reports/{group}/{name}/report.json"
            )
            rows.append(
                {
                    "name": name.upper(),
                    "error": float(report["error_pct"]),
                    "measured": float(report["measured"]["train_iteration_time_ms"]),
                    "estimated": float(report["estimate"]["train_iteration_time_ms"]),
                }
            )
    return rows


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def bar_chart_svg(
    title: str,
    subtitle: str,
    rows: list[dict],
    value_key: str,
    unit: str,
    threshold: float | None,
    color: str,
    output_path: Path,
) -> None:
    width = 1120
    height = 560
    margin_left = 180
    margin_right = 50
    margin_top = 110
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(row[value_key] for row in rows)
    if threshold is not None:
        max_value = max(max_value, threshold)
    max_value *= 1.15 if max_value else 1.0
    band = plot_height / max(len(rows), 1)
    bar_height = min(34, band * 0.58)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f2e7"/>',
        '<rect x="18" y="18" width="1084" height="524" rx="26" fill="#fffaf1" stroke="#d5c8b3" stroke-width="2"/>',
        f'<text x="{margin_left}" y="56" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="#1e1a16">{svg_escape(title)}</text>',
        f'<text x="{margin_left}" y="84" font-family="Arial, sans-serif" font-size="15" fill="#6d6457">{svg_escape(subtitle)}</text>',
    ]

    for tick_index in range(6):
        tick_value = max_value * tick_index / 5
        x = margin_left + plot_width * tick_index / 5
        lines.append(
            f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#e7ddcc" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#7d7468">{tick_value:.1f}{unit}</text>'
        )

    if threshold is not None:
        threshold_x = margin_left + (threshold / max_value) * plot_width
        lines.append(
            f'<line x1="{threshold_x:.1f}" y1="{margin_top - 8}" x2="{threshold_x:.1f}" y2="{height - margin_bottom}" stroke="#b24a2c" stroke-width="2.5" stroke-dasharray="8 6"/>'
        )
        lines.append(
            f'<text x="{threshold_x + 8:.1f}" y="{margin_top - 14}" font-family="Arial, sans-serif" font-size="12" fill="#b24a2c">20% threshold</text>'
        )

    for index, row in enumerate(rows):
        y = margin_top + band * index + (band - bar_height) / 2
        bar_width = (row[value_key] / max_value) * plot_width if max_value else 0
        lines.append(
            f'<text x="{margin_left - 14}" y="{y + bar_height / 2 + 5:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#2a241d">{svg_escape(row["name"])}</text>'
        )
        lines.append(
            f'<rect x="{margin_left}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="10" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{margin_left + bar_width + 10:.1f}" y="{y + bar_height / 2 + 5:.1f}" font-family="Arial, sans-serif" font-size="14" fill="#3d352c">{row[value_key]:.4f}{unit}</text>'
        )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def grouped_compare_svg(
    title: str,
    subtitle: str,
    rows: list[dict],
    output_path: Path,
) -> None:
    width = 1180
    height = 620
    margin_left = 120
    margin_right = 50
    margin_top = 120
    margin_bottom = 110
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(max(row["measured"], row["estimated"]) for row in rows) * 1.18
    group_width = plot_width / max(len(rows), 1)
    bar_width = min(42, group_width * 0.28)
    measured_color = "#1d6b52"
    estimated_color = "#c96d2d"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f2e7"/>',
        '<rect x="18" y="18" width="1144" height="584" rx="26" fill="#fffaf1" stroke="#d5c8b3" stroke-width="2"/>',
        f'<text x="{margin_left}" y="58" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="#1e1a16">{svg_escape(title)}</text>',
        f'<text x="{margin_left}" y="86" font-family="Arial, sans-serif" font-size="15" fill="#6d6457">{svg_escape(subtitle)}</text>',
        f'<rect x="{width - 250}" y="42" width="14" height="14" rx="4" fill="{measured_color}"/>',
        f'<text x="{width - 228}" y="54" font-family="Arial, sans-serif" font-size="13" fill="#4d4439">Measured</text>',
        f'<rect x="{width - 150}" y="42" width="14" height="14" rx="4" fill="{estimated_color}"/>',
        f'<text x="{width - 128}" y="54" font-family="Arial, sans-serif" font-size="13" fill="#4d4439">Estimated</text>',
    ]

    for tick_index in range(6):
        tick_value = max_value * tick_index / 5
        y = height - margin_bottom - plot_height * tick_index / 5
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e7ddcc" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#7d7468">{tick_value:.0f} ms</text>'
        )

    for index, row in enumerate(rows):
        center_x = margin_left + group_width * index + group_width / 2
        measured_h = plot_height * row["measured"] / max_value if max_value else 0
        estimated_h = plot_height * row["estimated"] / max_value if max_value else 0
        measured_x = center_x - bar_width - 6
        estimated_x = center_x + 6
        measured_y = height - margin_bottom - measured_h
        estimated_y = height - margin_bottom - estimated_h
        lines.append(
            f'<rect x="{measured_x:.1f}" y="{measured_y:.1f}" width="{bar_width:.1f}" height="{measured_h:.1f}" rx="8" fill="{measured_color}"/>'
        )
        lines.append(
            f'<rect x="{estimated_x:.1f}" y="{estimated_y:.1f}" width="{bar_width:.1f}" height="{estimated_h:.1f}" rx="8" fill="{estimated_color}"/>'
        )
        lines.append(
            f'<text x="{center_x:.1f}" y="{height - margin_bottom + 26}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#2a241d">{svg_escape(row["name"])}</text>'
        )
        lines.append(
            f'<text x="{center_x:.1f}" y="{height - margin_bottom + 46}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#7d7468">err {row["error"]:.2f}%</text>'
        )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class RasterCanvas:
    def __init__(self, width: int, height: int, bg: tuple[int, int, int]) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(bg * width * height)

    def _set(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
        index = (y * self.width + x) * 3
        self.pixels[index:index + 3] = bytes(color)

    def rect(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        color: tuple[int, int, int],
    ) -> None:
        for yy in range(y, y + h):
            if yy < 0 or yy >= self.height:
                continue
            for xx in range(x, x + w):
                self._set(xx, yy, color)

    def line(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        while True:
            self.rect(x1 - thickness // 2, y1 - thickness // 2, thickness, thickness, color)
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy

    def text(
        self,
        x: int,
        y: int,
        text: str,
        color: tuple[int, int, int],
        scale: int = 2,
    ) -> None:
        cursor_x = x
        for char in text.upper():
            glyph = FONT_5X7.get(char, FONT_5X7[" "])
            for row_index, row in enumerate(glyph):
                for col_index, bit in enumerate(row):
                    if bit == "1":
                        self.rect(
                            cursor_x + col_index * scale,
                            y + row_index * scale,
                            scale,
                            scale,
                            color,
                        )
            cursor_x += 6 * scale

    def save_png(self, output_path: Path) -> None:
        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack("!I", len(data))
                + tag
                + data
                + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        raw = bytearray()
        row_bytes = self.width * 3
        for y in range(self.height):
            raw.append(0)
            start = y * row_bytes
            raw.extend(self.pixels[start:start + row_bytes])
        png = bytearray(b"\x89PNG\r\n\x1a\n")
        png.extend(
            chunk(
                b"IHDR",
                struct.pack("!IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0),
            )
        )
        png.extend(chunk(b"IDAT", zlib.compress(bytes(raw), level=9)))
        png.extend(chunk(b"IEND", b""))
        output_path.write_bytes(bytes(png))


def bar_chart_png(
    title: str,
    rows: list[dict],
    value_key: str,
    threshold: float | None,
    color: tuple[int, int, int],
    output_path: Path,
) -> None:
    width, height = 1200, 620
    margin_left, margin_right, margin_top, margin_bottom = 220, 70, 120, 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(row[value_key] for row in rows)
    if threshold is not None:
        max_value = max(max_value, threshold)
    max_value *= 1.15 if max_value else 1.0
    band = plot_height / max(len(rows), 1)
    bar_height = int(min(34, band * 0.55))
    canvas = RasterCanvas(width, height, PALETTE["bg"])
    canvas.rect(18, 18, width - 36, height - 36, PALETTE["panel"])
    canvas.text(70, 38, title, PALETTE["text"], scale=3)
    canvas.text(70, 76, "REAL LLAMA-3.1-8B ON MLU580", PALETTE["muted"], scale=2)
    for tick_index in range(6):
        tick_value = max_value * tick_index / 5
        x = int(margin_left + plot_width * tick_index / 5)
        canvas.line(x, margin_top, x, height - margin_bottom, PALETTE["grid"])
        canvas.text(x - 10, height - margin_bottom + 18, f"{tick_value:.1f}", PALETTE["muted"], scale=1)
    if threshold is not None:
        x = int(margin_left + (threshold / max_value) * plot_width)
        for y in range(margin_top, height - margin_bottom, 10):
            canvas.line(x, y, x, min(y + 5, height - margin_bottom), PALETTE["red"], thickness=2)
        canvas.text(x + 10, margin_top - 26, "20% THRESHOLD", PALETTE["red"], scale=1)
    for index, row in enumerate(rows):
        y = int(margin_top + band * index + (band - bar_height) / 2)
        bar_width = int((row[value_key] / max_value) * plot_width) if max_value else 0
        canvas.text(24, y + 8, row["name"], PALETTE["text"], scale=2)
        canvas.rect(margin_left, y, bar_width, bar_height, color)
        canvas.text(margin_left + bar_width + 12, y + 9, f"{row[value_key]:.2f}%", PALETTE["muted"], scale=2)
    canvas.save_png(output_path)


def grouped_compare_png(
    title: str,
    rows: list[dict],
    output_path: Path,
) -> None:
    width, height = 1280, 680
    margin_left, margin_right, margin_top, margin_bottom = 100, 50, 130, 110
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(max(row["measured"], row["estimated"]) for row in rows) * 1.18
    group_width = plot_width / max(len(rows), 1)
    bar_width = int(min(40, group_width * 0.25))
    canvas = RasterCanvas(width, height, PALETTE["bg"])
    canvas.rect(18, 18, width - 36, height - 36, PALETTE["panel"])
    canvas.text(70, 40, title, PALETTE["text"], scale=3)
    canvas.text(70, 78, "MEASURED VS ESTIMATED", PALETTE["muted"], scale=2)
    canvas.rect(width - 255, 42, 18, 18, PALETTE["green"])
    canvas.text(width - 228, 44, "MEASURED", PALETTE["text"], scale=2)
    canvas.rect(width - 118, 42, 18, 18, PALETTE["orange"])
    canvas.text(width - 90, 44, "EST", PALETTE["text"], scale=2)
    for tick_index in range(6):
        tick_value = max_value * tick_index / 5
        y = int(height - margin_bottom - plot_height * tick_index / 5)
        canvas.line(margin_left, y, width - margin_right, y, PALETTE["grid"])
        canvas.text(10, y - 6, f"{tick_value:.0f}", PALETTE["muted"], scale=2)
    for index, row in enumerate(rows):
        center_x = int(margin_left + group_width * index + group_width / 2)
        measured_h = int(plot_height * row["measured"] / max_value) if max_value else 0
        estimated_h = int(plot_height * row["estimated"] / max_value) if max_value else 0
        mx = center_x - bar_width - 6
        ex = center_x + 6
        my = height - margin_bottom - measured_h
        ey = height - margin_bottom - estimated_h
        canvas.rect(mx, my, bar_width, measured_h, PALETTE["green"])
        canvas.rect(ex, ey, bar_width, estimated_h, PALETTE["orange"])
        canvas.text(center_x - 28, height - margin_bottom + 22, row["name"], PALETTE["text"], scale=2)
        canvas.text(center_x - 18, height - margin_bottom + 46, f"{row['error']:.1f}%", PALETTE["muted"], scale=1)
    canvas.save_png(output_path)


def main() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    infer = inference_rows()
    train = training_rows()
    bar_chart_svg(
        title="Inference Relative Error",
        subtitle="Real Llama-3.1-8B on Cambricon MLU580, request-level validation",
        rows=infer,
        value_key="error",
        unit="%",
        threshold=20.0,
        color="#1d6b52",
        output_path=CHART_DIR / "inference_error.svg",
    )
    bar_chart_svg(
        title="Training Relative Error",
        subtitle="Real Llama-3.1-8B on Cambricon MLU580, iteration-level validation",
        rows=train,
        value_key="error",
        unit="%",
        threshold=20.0,
        color="#8b3d21",
        output_path=CHART_DIR / "training_error.svg",
    )
    grouped_compare_svg(
        title="Inference Measured vs Estimated Time",
        subtitle="Measured request time against simulated request makespan",
        rows=infer,
        output_path=CHART_DIR / "inference_measured_vs_estimated.svg",
    )
    grouped_compare_svg(
        title="Training Measured vs Estimated Time",
        subtitle="Measured train iteration time against simulated iteration time",
        rows=train,
        output_path=CHART_DIR / "training_measured_vs_estimated.svg",
    )
    bar_chart_png(
        title="INFERENCE ERROR (%)",
        rows=infer,
        value_key="error",
        threshold=20.0,
        color=PALETTE["green"],
        output_path=CHART_DIR / "inference_error.png",
    )
    bar_chart_png(
        title="TRAINING ERROR (%)",
        rows=train,
        value_key="error",
        threshold=20.0,
        color=PALETTE["brown"],
        output_path=CHART_DIR / "training_error.png",
    )
    grouped_compare_png(
        title="INFERENCE TIME (MS)",
        rows=infer,
        output_path=CHART_DIR / "inference_measured_vs_estimated.png",
    )
    grouped_compare_png(
        title="TRAINING TIME (MS)",
        rows=train,
        output_path=CHART_DIR / "training_measured_vs_estimated.png",
    )


if __name__ == "__main__":
    main()
