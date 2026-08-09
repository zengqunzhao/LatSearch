#!/usr/bin/env python3
"""Generate videos and intermediate latents for latent-reward training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from latsearch.cli_utils import (
    cuda_device_index,
    load_prompts,
    require_path,
    save_video,
    seed_everything,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect (prompt, latent, timestep, reward, similarity) training tuples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prompt-file",
        default="prompts/LatSearch_train_prompts.json",
        help="JSON prompt list used to construct the latent-reward dataset.",
    )
    parser.add_argument("--model-name", choices=("t2v-1.3B", "t2v-14B"), default="t2v-1.3B")
    parser.add_argument("--model-path", default="checkpoints/Wan2.1-T2V-1.3B")
    parser.add_argument("--video-reward-path", default="checkpoints/VideoReward")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--seeds", nargs="+", type=int, default=[200, 300, 400, 500, 600])
    parser.add_argument("--selected-steps", nargs="+", type=int, default=[10, 15, 20, 25, 30])
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.selected_steps != sorted(set(args.selected_steps)):
        raise ValueError("--selected-steps must contain unique, increasing indices.")
    if args.selected_steps[-1] >= args.sampling_steps:
        raise ValueError("Selected steps must be smaller than --sampling-steps.")
    if (args.num_frames - 1) % 4:
        raise ValueError("--num-frames must have the form 4n+1.")

    model_path = require_path(args.model_path, "Wan checkpoint")
    video_reward_path = require_path(args.video_reward_path, "VideoReward checkpoint")
    prompts = load_prompts(None, args.prompt_file, max_prompts=args.max_prompts)
    device_id = cuda_device_index(args.device)

    import torch

    from third_party.VideoReward.score import VideoVLMRewardInference
    from third_party.WanVideoModel import wan
    from third_party.WanVideoModel.wan.configs import WAN_CONFIGS

    pipeline = wan.WanT2V(
        config=WAN_CONFIGS[args.model_name],
        checkpoint_dir=str(model_path),
        device_id=device_id,
    )
    verifier = VideoVLMRewardInference(
        str(video_reward_path),
        device=torch.device(args.device),
        dtype=torch.float16,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        seed_everything(seed)
        seed_dir = output_dir / f"seed_{seed}"
        metadata_path = metadata_dir / f"all_latent_metadata_seed_{seed:04d}.json"
        records = []
        if metadata_path.exists():
            with metadata_path.open(encoding="utf-8") as handle:
                records = json.load(handle)

        for prompt_index, item in enumerate(prompts[len(records) :], start=len(records)):
            sample_seed = seed + prompt_index
            video, latents, similarities = pipeline.generate_with_latent(
                input_prompt=item.text,
                size=(args.width, args.height),
                frame_num=args.num_frames,
                shift=args.flow_shift,
                sample_solver="unipc",
                sampling_steps=args.sampling_steps,
                guide_scale=args.guidance_scale,
                seed=sample_seed,
                selected_timestep_keys=args.selected_steps,
            )
            sample_path = seed_dir / f"{prompt_index:04d}"
            sample_path.mkdir(parents=True, exist_ok=True)
            video_path = sample_path / "video.mp4"
            save_video(video, video_path, args.fps)

            latent_paths = {}
            latent_similarities = {}
            for step in args.selected_steps:
                latent_path = sample_path / f"latent_step_t{step}_tensor.pt"
                torch.save(latents[step].cpu(), latent_path)
                latent_paths[f"t{step}"] = os.path.relpath(latent_path, metadata_dir)
                # Equation (7) in the paper rescales cosine similarity to [0, 1].
                latent_similarities[f"t{step}"] = 0.5 * (1.0 + similarities[step])

            rewards = verifier.reward(
                video[None].permute(0, 2, 1, 3, 4),
                [item.text],
                use_norm=False,
            )[0]
            records.append(
                {
                    "prompt": item.text,
                    "video_path": os.path.relpath(video_path, metadata_dir),
                    "output_reward": [rewards],
                    "latent_tensor_path": [latent_paths],
                    "latent_z0_similarity": [latent_similarities],
                }
            )
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(records, handle, indent=2)
            print(f"Collected seed={seed} prompt={prompt_index}: {item.text}")


if __name__ == "__main__":
    main()
