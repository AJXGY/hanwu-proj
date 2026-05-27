#!/usr/bin/env python3
"""Parse Cambricon CNCLBenchmark text logs into a normalized CSV table.

This script extracts latency rows from the official benchmark output so they
can be analyzed by the model fitting and plotting tools.
"""

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse CNCL benchmark text logs.")
    parser.add_argument("--input", required=True, help="Input log path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--operator", required=True, help="Operator name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = []
    for line in input_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        tokens = line.split()
        if len(tokens) < 4:
            continue
        if not tokens[0].isdigit() or not tokens[1].isdigit():
            continue
        latency_token = None
        for token in tokens[3:]:
            if token.count(".") == 1:
                left, right = token.split(".", 1)
                if left.isdigit() and right.isdigit():
                    latency_token = token
                    break
        if latency_token is None:
            continue
        rows.append(
            {
                "operator": args.operator,
                "message_bytes": int(tokens[0]),
                "avg_ms": float(latency_token) / 1000.0,
                "min_ms": float(latency_token) / 1000.0,
                "max_ms": float(latency_token) / 1000.0,
                "std_ms": 0.0,
                "world_size": 2,
                "device_type": "MLU580",
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "operator",
                "message_bytes",
                "avg_ms",
                "min_ms",
                "max_ms",
                "std_ms",
                "world_size",
                "device_type",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {len(rows)} rows into {output_path}")


if __name__ == "__main__":
    main()
