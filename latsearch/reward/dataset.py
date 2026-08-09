"""Dataset for latent-reward training metadata."""

from __future__ import annotations

import json
import random
import unicodedata
from pathlib import Path

import torch
from torch.utils.data import Dataset

STEP_TO_TIMESTEP = {
    "t10": 200,
    "t15": 300,
    "t20": 400,
    "t25": 500,
    "t30": 600,
}


class LatentJsonDataLoader(Dataset):
    """Load latent tensors using a deterministic prompt-level split."""

    def __init__(
        self,
        json_files_root: str | Path,
        data_mode: str,
        val_step_chose: str | None,
        train_ratio: float = 0.8,
        split_seed: int = 1203,
    ) -> None:
        if data_mode not in {"train", "validation"}:
            raise ValueError("data_mode must be 'train' or 'validation'.")
        if data_mode == "validation" and val_step_chose not in STEP_TO_TIMESTEP:
            raise ValueError(f"Validation step must be one of {sorted(STEP_TO_TIMESTEP)}.")
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1.")

        self.json_files_root = Path(json_files_root).expanduser().resolve()
        metadata_files = sorted(self.json_files_root.glob("all_latent_metadata_seed_*.json"))
        if not metadata_files:
            raise ValueError(f"No seed metadata files found in {self.json_files_root}.")

        self.data_mode = data_mode
        self.val_step_chose = val_step_chose
        all_records = []
        for metadata_path in metadata_files:
            with metadata_path.open(encoding="utf-8") as handle:
                records = json.load(handle)
            if not isinstance(records, list):
                raise ValueError(f"Metadata must contain a JSON list: {metadata_path}")
            all_records.extend(records)

        prompt_ids = sorted({_canonical_prompt(record["prompt"]) for record in all_records})
        if len(prompt_ids) < 2:
            raise ValueError("At least two unique prompts are required for an 80/20 split.")
        random.Random(split_seed).shuffle(prompt_ids)
        split_index = min(len(prompt_ids) - 1, max(1, int(train_ratio * len(prompt_ids))))
        selected_prompt_ids = set(
            prompt_ids[:split_index] if data_mode == "train" else prompt_ids[split_index:]
        )
        self.prompt_ids = frozenset(selected_prompt_ids)
        self.records = [
            record
            for record in all_records
            if _canonical_prompt(record["prompt"]) in selected_prompt_ids
        ]

        if not self.records:
            raise ValueError(f"The {data_mode} split contains no samples.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        selected_step = (
            random.choice(tuple(STEP_TO_TIMESTEP))
            if self.data_mode == "train"
            else self.val_step_chose
        )
        assert selected_step is not None

        latent_path = Path(record["latent_tensor_path"][0][selected_step]).expanduser()
        if not latent_path.is_absolute():
            latent_path = self.json_files_root / latent_path
        try:
            latent = torch.load(latent_path, map_location="cpu", weights_only=True)
        except TypeError:
            latent = torch.load(latent_path, map_location="cpu")
        latent = latent.permute(1, 0, 2, 3)

        rewards = record["output_reward"][0]
        similarity = record["latent_z0_similarity"][0][selected_step]
        return (
            latent,
            record["prompt"],
            rewards["VQ"],
            rewards["MQ"],
            rewards["TA"],
            similarity,
            STEP_TO_TIMESTEP[selected_step],
        )


def _canonical_prompt(prompt: str) -> str:
    """Normalize prompt identity without changing the text passed to the model."""

    return " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())
