from __future__ import annotations

import argparse
from pathlib import Path

from train0411_clj.smoke_model import build_model, build_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny local Llama smoke model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        print(f"Smoke model already exists at {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer()
    model = build_model()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Wrote smoke model to {output_dir}")


if __name__ == "__main__":
    main()
