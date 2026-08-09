#!/usr/bin/env python3
"""Inspect latent-reward predictions on saved latent metadata."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from latsearch.cli_utils import load_torch_state_dict, require_path, seed_everything

STEP_TO_TIMESTEP = {"t10": 200, "t15": 300, "t20": 400, "t25": 500, "t30": 600}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a LatSearch latent reward checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--latent-reward-checkpoint", required=True)
    parser.add_argument("--reward-backbone-path", default="checkpoints/VideoReward")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--step", choices=tuple(STEP_TO_TIMESTEP), default="t20")
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=1203)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    import torch

    from latsearch.reward.checkpoint import load_latent_reward_state
    from latsearch.reward.model import LatentReward

    device = torch.device(args.device)
    model = LatentReward(
        load_from_pretrained=str(require_path(args.reward_backbone_path, "Reward backbone")),
        device=device,
        dtype=torch.bfloat16,
    ).to(device)
    load_latent_reward_state(model, load_torch_state_dict(args.latent_reward_checkpoint))
    model.eval()

    metadata_path = require_path(args.metadata, "Metadata file")
    with metadata_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    for record in records:
        latent_path = Path(record["latent_tensor_path"][0][args.step]).expanduser()
        if not latent_path.is_absolute():
            latent_path = metadata_path.parent / latent_path
        latent = torch.load(latent_path, map_location="cpu").permute(1, 0, 2, 3)
        latent = latent.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        denoising_step = torch.tensor([STEP_TO_TIMESTEP[args.step]], device=device)

        started = time.perf_counter()
        with torch.no_grad():
            batch = model.prepare_batch(
                videos=[latent[0]],
                prompts=[record["prompt"]],
                denoising_steps=denoising_step,
                num_frames=args.num_frames,
            )
            prediction = model(batch)[0].float().cpu().tolist()
        elapsed = time.perf_counter() - started
        target = record["output_reward"][0]
        print(
            f"prediction VQ={prediction[0]:.4f} MQ={prediction[1]:.4f} "
            f"TA={prediction[2]:.4f} | target VQ={target['VQ']:.4f} "
            f"MQ={target['MQ']:.4f} TA={target['TA']:.4f} | {elapsed:.3f}s"
        )


if __name__ == "__main__":
    main()
