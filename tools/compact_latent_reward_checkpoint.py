#!/usr/bin/env python3
"""Convert a legacy full latent-reward checkpoint to the compact format."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latsearch.cli_utils import load_torch_state_dict
from latsearch.reward.checkpoint import compact_legacy_state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Legacy full latent-reward checkpoint.")
    parser.add_argument("output", help="Destination for the compact checkpoint.")
    args = parser.parse_args()

    state = compact_legacy_state_dict(load_torch_state_dict(args.input))
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)
    size_gib = output_path.stat().st_size / 1024**3
    print(f"Saved {len(state)} tensors to {output_path} ({size_gib:.2f} GiB)")


if __name__ == "__main__":
    main()
