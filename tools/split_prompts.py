#!/usr/bin/env python3
"""Split a JSON prompt list into approximately equal parts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON file containing a list of prompts.")
    parser.add_argument("--parts", type=int, default=5, help="Number of output files.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to the input directory.")
    args = parser.parse_args()

    if args.parts < 1:
        parser.error("--parts must be at least 1")

    with args.input.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Prompt file must contain a JSON list: {args.input}")

    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = math.ceil(len(records) / args.parts)
    for index in range(args.parts):
        part = records[index * chunk_size : (index + 1) * chunk_size]
        output_path = output_dir / f"{args.input.stem}_part{index + 1}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(part, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"Saved {len(part)} prompts to {output_path}")


if __name__ == "__main__":
    main()
