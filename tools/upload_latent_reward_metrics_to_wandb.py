#!/usr/bin/env python3
"""Upload saved latent-reward metrics to reproducible W&B runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import wandb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_root", type=Path, help="Directory containing seed_* runs.")
    parser.add_argument("--project", default="LatSearch")
    parser.add_argument("--entity")
    parser.add_argument("--group", help="W&B run group; defaults to the training parent name.")
    parser.add_argument("--job-type", default="latent-reward-training")
    parser.add_argument(
        "--mode",
        choices=("online", "offline"),
        default="online",
        help="W&B upload mode.",
    )
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    metrics = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            metrics.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    if not metrics:
        raise ValueError(f"No metrics found in {path}.")
    return sorted(metrics, key=lambda item: int(item["epoch"]))


def _find_manifest(training_root: Path) -> dict[str, Any]:
    candidates = (
        training_root / "run_manifest.json",
        training_root.parent / "run_manifest.json",
        training_root.parent.parent / "run_manifest.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return _load_json(candidate)
    return {}


def _stable_run_id(group: str, run_name: str) -> str:
    raw = f"{group}-{run_name}".lower()
    slug = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-")
    if len(slug) <= 96:
        return slug
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{slug[:83]}-{digest}"


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    epoch = int(metrics["epoch"])
    payload: dict[str, Any] = {"epoch": epoch}
    for name, value in metrics["train"].items():
        payload[f"train/{name}"] = value
    for step, scores in metrics["validation"].items():
        step_number = step.removeprefix("t")
        for dimension, value in scores.items():
            payload[f"val/pair_acc_{dimension}_step{step_number}"] = value
    payload["val/mean_validation_accuracy"] = metrics["mean_validation_accuracy"]
    return payload


def _seed_from_run_dir(run_dir: Path) -> int:
    match = re.fullmatch(r"seed_(\d+)", run_dir.name)
    if match is None:
        raise ValueError(f"Expected a seed_<number> directory, got {run_dir}.")
    return int(match.group(1))


def main() -> None:
    args = build_parser().parse_args()
    training_root = args.training_root.expanduser().resolve()
    run_dirs = sorted(path for path in training_root.glob("seed_*") if path.is_dir())
    if not run_dirs:
        raise ValueError(f"No seed_* runs found in {training_root}.")

    manifest = _find_manifest(training_root)
    shared_config = dict(manifest.get("configuration", {}))
    group = args.group or training_root.parent.name
    settings = wandb.Settings(
        disable_code=True,
        disable_git=True,
        save_code=False,
        console="off",
        x_disable_meta=True,
        x_disable_stats=True,
        x_save_requirements=False,
        ignore_globs=("*",),
        # Some managed clusters allow the W&B API but block its object-storage
        # domain. Scalar history still works; do not hang indefinitely on the
        # optional file uploads in that environment.
        x_file_transfer_retry_max=1,
        x_file_transfer_retry_wait_min_seconds=0,
        x_file_transfer_retry_wait_max_seconds=0,
        x_file_transfer_timeout_seconds=2,
    )

    for run_dir in run_dirs:
        metrics_path = run_dir / "latent_reward.metrics.jsonl"
        if not metrics_path.is_file():
            print(f"Skipping {run_dir.name}: metrics file does not exist yet.")
            continue

        seed = _seed_from_run_dir(run_dir)
        metrics = _load_metrics(metrics_path)
        run_config = {
            **shared_config,
            "training_seed": seed,
            "metrics_source": str(metrics_path),
        }
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            group=group,
            job_type=args.job_type,
            name=f"latent-reward-seed-{seed}",
            id=_stable_run_id(group, run_dir.name),
            resume="allow",
            mode=args.mode,
            config=run_config,
            tags=("latent-reward", "backfilled"),
            reinit="finish_previous",
            settings=settings,
        )
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")
        run.define_metric("val/mean_validation_accuracy", summary="max")

        uploaded_epochs = {int(epoch) for epoch in run.summary.get("uploaded_epochs", [])}
        for epoch_metrics in metrics:
            epoch = int(epoch_metrics["epoch"])
            if epoch not in uploaded_epochs:
                run.log(_flatten_metrics(epoch_metrics), step=epoch)
                uploaded_epochs.add(epoch)

        summary_path = run_dir / "latent_reward.summary.json"
        if summary_path.is_file():
            summary = _load_json(summary_path)
            run.summary["best_epoch"] = int(summary["best_epoch"])
            run.summary["best_mean_validation_accuracy"] = float(
                summary["best_mean_validation_accuracy"]
            )
        run.summary["uploaded_epochs"] = sorted(uploaded_epochs)
        run.summary["epochs_uploaded"] = len(uploaded_epochs)
        run.summary["selection_metric"] = "val/mean_validation_accuracy"
        print(f"{run_dir.name}: uploaded through epoch {max(uploaded_epochs)} -> {run.url}")
        run.finish()


if __name__ == "__main__":
    main()
