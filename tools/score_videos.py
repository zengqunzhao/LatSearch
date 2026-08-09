#!/usr/bin/env python3
"""Score generated MP4 files with VideoReward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score videos with VideoReward.")
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--video-reward-path", default="checkpoints/VideoReward")
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt", help="Use one prompt for every video.")
    prompts.add_argument(
        "--prompt-map",
        type=Path,
        help="JSON object mapping video filenames to prompts.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("video_rewards.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.video_dir.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {args.video_dir}")

    import torch
    from torchvision import io

    from third_party.VideoReward.score import VideoVLMRewardInference

    prompt_map = None
    if args.prompt_map:
        with args.prompt_map.open(encoding="utf-8") as handle:
            prompt_map = json.load(handle)
        if not isinstance(prompt_map, dict):
            raise ValueError("--prompt-map must contain a JSON object.")

    verifier = VideoVLMRewardInference(
        args.video_reward_path,
        device=torch.device(args.device),
        dtype=torch.float16,
    )
    results = []
    for video_path in sorted(args.video_dir.glob("*.mp4")):
        prompt = args.prompt if prompt_map is None else prompt_map.get(video_path.name)
        if not prompt:
            raise KeyError(f"No prompt provided for {video_path.name}")
        video, _, _ = io.read_video(str(video_path), output_format="TCHW")
        video = (video.float() / 127.5 - 1).unsqueeze(0)
        reward = verifier.reward(video, [prompt], use_norm=False)[0]
        results.append({"video": video_path.name, "prompt": prompt, "reward": reward})
        print(f"{video_path.name}: {reward}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
