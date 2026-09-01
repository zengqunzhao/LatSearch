#!/usr/bin/env python3
"""Validate latent-reward metadata and its referenced artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch

EXPECTED_STEPS = ("t10", "t15", "t20", "t25", "t30")
SEED_PATTERN = re.compile(r"all_latent_metadata_seed_(\d+)\.json$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_dir", type=Path)
    parser.add_argument("--expected-seeds", nargs="+", type=int)
    parser.add_argument("--expected-records-per-seed", type=int)
    parser.add_argument("--expected-shape", nargs=4, type=int, default=(16, 9, 60, 104))
    parser.add_argument(
        "--check-tensors",
        action="store_true",
        help="Load every latent tensor and check its shape and finite values.",
    )
    return parser


def _resolve(metadata_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (metadata_path.parent / path).resolve()


def _finite_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}.")
    return float(value)


def main() -> None:
    args = build_parser().parse_args()
    metadata_dir = args.metadata_dir.expanduser().resolve()
    metadata_files = sorted(metadata_dir.glob("all_latent_metadata_seed_*.json"))
    if not metadata_files:
        raise ValueError(f"No seed metadata files found in {metadata_dir}.")

    found_seeds = []
    reference_prompts = None
    total_records = 0
    total_latents = 0
    for metadata_path in metadata_files:
        match = SEED_PATTERN.fullmatch(metadata_path.name)
        if match is None:
            continue
        seed = int(match.group(1))
        found_seeds.append(seed)
        records = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"Metadata must contain a JSON list: {metadata_path}")
        if (
            args.expected_records_per_seed is not None
            and len(records) != args.expected_records_per_seed
        ):
            raise ValueError(
                f"{metadata_path} has {len(records)} records; "
                f"expected {args.expected_records_per_seed}."
            )

        prompts = []
        for index, record in enumerate(records):
            label = f"{metadata_path.name}[{index}]"
            prompt = record.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{label} has an invalid prompt.")
            prompts.append(prompt)

            video_path = _resolve(metadata_path, record["video_path"])
            if not video_path.is_file() or video_path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty video: {video_path}")

            rewards = record["output_reward"][0]
            for dimension in ("VQ", "MQ", "TA"):
                _finite_number(rewards[dimension], f"{label} reward {dimension}")

            latent_paths = record["latent_tensor_path"][0]
            similarities = record["latent_z0_similarity"][0]
            for step in EXPECTED_STEPS:
                latent_path = _resolve(metadata_path, latent_paths[step])
                if not latent_path.is_file() or latent_path.stat().st_size == 0:
                    raise FileNotFoundError(f"Missing or empty latent: {latent_path}")
                similarity = _finite_number(similarities[step], f"{label} similarity {step}")
                if not 0.0 <= similarity <= 1.0:
                    raise ValueError(f"{label} similarity {step} is outside [0, 1].")
                if args.check_tensors:
                    tensor = torch.load(latent_path, map_location="cpu", weights_only=True)
                    if tuple(tensor.shape) != tuple(args.expected_shape):
                        raise ValueError(
                            f"{latent_path} has shape {tuple(tensor.shape)}; "
                            f"expected {tuple(args.expected_shape)}."
                        )
                    if not torch.isfinite(tensor).all():
                        raise ValueError(f"{latent_path} contains non-finite values.")
                total_latents += 1

        if len(prompts) != len(set(prompts)):
            raise ValueError(f"Duplicate prompts found in {metadata_path}.")
        if reference_prompts is None:
            reference_prompts = prompts
        elif prompts != reference_prompts:
            raise ValueError(f"Prompt ordering differs in {metadata_path}.")
        total_records += len(records)
        print(f"seed={seed}: {len(records)} records OK")

    if args.expected_seeds is not None and sorted(found_seeds) != sorted(args.expected_seeds):
        raise ValueError(
            f"Found seeds {sorted(found_seeds)}; expected {sorted(args.expected_seeds)}."
        )

    print(
        f"Validated {len(found_seeds)} seed files, {total_records} records, "
        f"and {total_latents} latent references."
    )


if __name__ == "__main__":
    main()
