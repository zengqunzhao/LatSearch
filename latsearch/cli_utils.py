"""Shared command-line helpers for LatSearch and its evaluation baselines."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptItem:
    """One text prompt and its optional benchmark dimension."""

    text: str
    dimension: str | None = None


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible single-GPU inference."""

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cuda_device_index(device: str) -> int:
    """Return the CUDA index expected by the vendored Wan pipeline."""

    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", maxsplit=1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device: {device}") from exc
    raise ValueError("Wan inference requires a CUDA device such as 'cuda' or 'cuda:0'.")


def require_path(path: str | Path, label: str) -> Path:
    """Validate an input path and return its expanded absolute path."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _dimension(value: Any) -> str | None:
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value) if value else None


def load_prompts(
    prompt: str | None,
    prompt_file: str | None,
    dimensions: Iterable[str] | None = None,
    max_prompts: int | None = None,
) -> list[PromptItem]:
    """Load one prompt or a JSON prompt list used by VBench/VBench-2.0."""

    if bool(prompt) == bool(prompt_file):
        raise ValueError("Provide exactly one of --prompt or --prompt-file.")

    if prompt:
        items = [PromptItem(prompt.strip())]
    else:
        prompt_path = require_path(prompt_file or "", "Prompt file")
        with prompt_path.open(encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError("Prompt JSON must contain a list of strings or objects.")

        items = []
        for record in records:
            if isinstance(record, str):
                text, dimension = record, None
            elif isinstance(record, dict):
                text = record.get("prompt_en") or record.get("prompt")
                dimension = _dimension(record.get("dimension"))
            else:
                raise ValueError(f"Unsupported prompt record: {type(record).__name__}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "Every prompt record must contain non-empty 'prompt_en' or 'prompt'."
                )
            items.append(PromptItem(text.strip(), dimension))

    selected_dimensions = set(dimensions or [])
    if selected_dimensions:
        items = [item for item in items if item.dimension in selected_dimensions]
    if max_prompts is not None:
        if max_prompts < 1:
            raise ValueError("--max-prompts must be at least 1.")
        items = items[:max_prompts]
    if not items:
        raise ValueError("No prompts matched the requested dimensions.")
    return items


def samples_for_prompt(item: PromptItem, samples: int, diversity_samples: int) -> int:
    """Use the VBench-2.0 repetition count for the Diversity dimension."""

    return diversity_samples if item.dimension == "Diversity" else samples


def output_path(
    output_dir: str | Path,
    item: PromptItem,
    prompt_index: int,
    sample_index: int,
) -> Path:
    """Build a stable, filesystem-safe output path without exposing the full prompt."""

    dimension = _slug(item.dimension or "samples", 48)
    prompt_slug = _slug(item.text, 72)
    digest = hashlib.sha1(item.text.encode("utf-8")).hexdigest()[:8]
    directory = Path(output_dir).expanduser() / dimension
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{prompt_index:04d}_{prompt_slug}_{digest}_seed{sample_index}.mp4"


def _slug(value: str, max_length: int) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return (value or "sample")[:max_length]


def load_torch_state_dict(path: str | Path) -> dict[str, Any]:
    """Load a PyTorch state dict on CPU across supported PyTorch versions."""

    import torch

    checkpoint_path = require_path(path, "Latent reward checkpoint")
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint must contain a state dict: {checkpoint_path}")
    return state


def save_video(video: Any, path: Path, fps: int) -> None:
    """Write a generated Wan tensor to an MP4 file."""

    from third_party.WanVideoModel.wan.utils.utils import cache_video

    cache_video(
        tensor=video[None],
        save_file=str(path),
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
