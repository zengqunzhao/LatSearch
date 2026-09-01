"""Compact checkpoint helpers for the latent reward model."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn


def _visual_key_suffix(name: str) -> str | None:
    if name.startswith("visual."):
        return name
    marker = ".visual."
    if marker in name:
        return f"visual.{name.split(marker, 1)[1]}"
    return None


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return CPU copies of parameters optimized by latent-reward training."""

    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise RuntimeError("The latent reward model has no trainable parameters.")
    return state


def compact_legacy_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Extract trainable visual-tower tensors from a full checkpoint."""

    compact = {
        name: tensor
        for name, tensor in state.items()
        if name.startswith("visual.") or ".visual." in name
    }
    if not any(name.startswith("visual.") or ".visual." in name for name in compact):
        raise RuntimeError("The input does not contain latent-reward visual-tower tensors.")
    return compact


def save_trainable_checkpoint(model: nn.Module, path: str | Path) -> Path:
    """Atomically save only trainable latent-reward parameters."""

    checkpoint_path = Path(path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp")
    state = trainable_state_dict(model)
    torch.save(state, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    size_gib = checkpoint_path.stat().st_size / 1024**3
    print(f"Saved {len(state)} trainable tensors to {checkpoint_path} ({size_gib:.2f} GiB)")
    return checkpoint_path


def load_latent_reward_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load a compact or legacy full checkpoint and validate trainable keys."""

    model_state = model.state_dict()
    visual_targets = {
        suffix: name for name in model_state if (suffix := _visual_key_suffix(name)) is not None
    }
    remapped = {}
    unmatched_visual = []
    for name, tensor in state.items():
        target_name = name if name in model_state else None
        suffix = _visual_key_suffix(name)
        if target_name is None and suffix is not None:
            target_name = visual_targets.get(suffix)
        if target_name is None:
            if suffix is not None:
                unmatched_visual.append(name)
            continue
        if target_name in remapped:
            raise RuntimeError(f"Duplicate latent reward checkpoint key: {target_name}")
        remapped[target_name] = tensor

    if unmatched_visual:
        keys = ", ".join(unmatched_visual[:5])
        raise RuntimeError(f"Unexpected latent reward visual keys: {keys}")

    incompatible = model.load_state_dict(remapped, strict=False)

    required = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing_required = required.intersection(incompatible.missing_keys)
    if missing_required:
        keys = ", ".join(sorted(missing_required)[:5])
        raise RuntimeError(f"Latent reward checkpoint is missing trainable keys: {keys}")

    print(f"Loaded {len(remapped)} latent reward tensors")
