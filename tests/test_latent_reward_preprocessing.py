from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from latsearch.reward.model import LatentReward


class _Processor:
    def __init__(self) -> None:
        self.videos = None

    def apply_chat_template(self, chat_data, **_kwargs):
        return ["prompt"] * len(chat_data)

    def __call__(self, *, videos, **_kwargs):
        self.videos = videos
        return {"input_ids": torch.ones(len(videos), 1, dtype=torch.long)}


class _StepEmbedding(nn.Module):
    inputs: list[torch.Tensor] = []

    def __init__(self, _num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        self.inputs.append(steps.detach().cpu())
        return steps[:, None].expand(-1, self.embedding_dim).float()


class LatentRewardPreprocessingTest(unittest.TestCase):
    def test_uses_supplied_timestep_without_registering_checkpoint_weights(self) -> None:
        model = LatentReward.__new__(LatentReward)
        nn.Module.__init__(model)
        model.device = torch.device("cpu")
        model.processor = _Processor()
        model.data_config = SimpleNamespace(
            fps=2.0,
            num_frames=2,
            max_frame_pixels=200704,
            sample_type="uniform",
            eval_dim="VQ",
            prompt_template_type="none",
        )
        model.latent_channels = 16

        videos = [torch.zeros(2, 16, 4, 4), torch.zeros(2, 16, 4, 4)]
        steps = torch.tensor([200, 300])
        _StepEmbedding.inputs.clear()
        with (
            patch(
                "latsearch.reward.model.process_vision_info",
                return_value=(None, videos),
            ),
            patch("latsearch.reward.model.nn.Embedding", _StepEmbedding),
        ):
            model.prepare_batch(videos, ["first", "second"], steps)

        self.assertFalse(hasattr(model, "step_embedding"))
        self.assertTrue(torch.equal(_StepEmbedding.inputs[0], steps))
        self.assertEqual(model.processor.videos[0][0, 0, 0, 0].item(), 200.0)
        self.assertEqual(model.processor.videos[1][0, 0, 0, 0].item(), 300.0)


if __name__ == "__main__":
    unittest.main()
