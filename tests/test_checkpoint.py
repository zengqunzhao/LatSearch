from __future__ import annotations

import unittest

import torch
from torch import nn

from latsearch.reward.checkpoint import (
    compact_legacy_state_dict,
    load_latent_reward_state,
)


class _LegacyReward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Linear(2, 2, bias=False)


class CheckpointCompatibilityTest(unittest.TestCase):
    def test_compacts_legacy_visual_only_checkpoint(self) -> None:
        state = {
            "model.visual.weight": torch.zeros(2, 2),
            "model.language.weight": torch.zeros(2, 2),
            "step_embedding.weight": torch.zeros(1000, 16),
        }

        self.assertEqual(list(compact_legacy_state_dict(state)), ["model.visual.weight"])

    def test_loader_accepts_matching_legacy_model(self) -> None:
        model = _LegacyReward()
        state = {"model.base_model.model.model.visual.weight": torch.ones_like(model.visual.weight)}

        load_latent_reward_state(model, state)
        self.assertTrue(
            torch.equal(
                model.visual.weight,
                state["model.base_model.model.model.visual.weight"],
            )
        )

    def test_compactor_rejects_checkpoint_without_visual_tower(self) -> None:
        state = {"model.language.weight": torch.zeros(2, 2)}

        with self.assertRaisesRegex(RuntimeError, "visual-tower"):
            compact_legacy_state_dict(state)


if __name__ == "__main__":
    unittest.main()
