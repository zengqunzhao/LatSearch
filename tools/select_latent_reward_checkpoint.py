#!/usr/bin/env python3
"""Select and package the best latent-reward training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_root", type=Path, help="Directory containing seed_* runs.")
    parser.add_argument("output_dir", type=Path, help="Directory for the selected checkpoint.")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = build_parser().parse_args()
    summaries = sorted(args.training_root.expanduser().glob("seed_*/latent_reward.summary.json"))
    if not summaries:
        raise ValueError(f"No training summaries found in {args.training_root}.")

    candidates = []
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint = Path(summary["checkpoint"]).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (summary_path.parent / checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint referenced by {summary_path}: {checkpoint}"
            )
        candidates.append(
            {
                "run": summary_path.parent.name,
                "best_epoch": int(summary["best_epoch"]),
                "mean_validation_accuracy": float(summary["best_mean_validation_accuracy"]),
                "checkpoint": checkpoint,
                "summary": summary_path.resolve(),
            }
        )

    best = max(candidates, key=lambda item: item["mean_validation_accuracy"])
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint = output_dir / "latent_reward.pt"
    shutil.copy2(best["checkpoint"], output_checkpoint)

    selection = {
        "selected_run": best["run"],
        "best_epoch": best["best_epoch"],
        "mean_validation_accuracy": best["mean_validation_accuracy"],
        "source_checkpoint": str(best["checkpoint"]),
        "checkpoint": str(output_checkpoint),
        "sha256": _sha256(output_checkpoint),
        "candidates": [
            {
                "run": item["run"],
                "best_epoch": item["best_epoch"],
                "mean_validation_accuracy": item["mean_validation_accuracy"],
                "summary": str(item["summary"]),
            }
            for item in candidates
        ],
    }
    selection_path = output_dir / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
