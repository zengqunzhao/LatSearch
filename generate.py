#!/usr/bin/env python3
"""Generate videos with LatSearch reward-guided resampling and pruning."""

from __future__ import annotations

import argparse

from latsearch.cli_utils import (
    cuda_device_index,
    load_prompts,
    load_torch_state_dict,
    output_path,
    require_path,
    samples_for_prompt,
    save_video,
    seed_everything,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate text-to-video samples with LatSearch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt", help="A single text prompt.")
    prompts.add_argument("--prompt-file", help="JSON prompt list (VBench format is supported).")
    parser.add_argument(
        "--dimensions", nargs="+", help="Only generate matching benchmark dimensions."
    )
    parser.add_argument("--max-prompts", type=int, help="Limit the number of selected prompts.")

    parser.add_argument("--model-name", choices=("t2v-1.3B", "t2v-14B"), default="t2v-1.3B")
    parser.add_argument("--model-path", default="checkpoints/Wan2.1-T2V-1.3B")
    parser.add_argument(
        "--reward-backbone-path",
        default="checkpoints/VideoReward",
        help="VideoReward directory containing model_config.json and checkpoint-*.",
    )
    parser.add_argument("--latent-reward-checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/latsearch")
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num-frames", type=int, default=33, help="Must be 4n+1 for Wan2.1.")
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--solver", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--diversity-samples", type=int, default=20)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--offload-model", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--search-schedule", nargs="+", type=int, default=[10, 15, 20])
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument("--noise-perturbation", type=float, default=0.1, dest="beta")
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_frames < 1 or (args.num_frames - 1) % 4:
        raise ValueError("--num-frames must have the form 4n+1 (for example, 33 or 81).")
    if args.num_candidates < 2:
        raise ValueError("--num-candidates must be at least 2.")
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("--noise-perturbation must be in [0, 1].")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if args.search_schedule != sorted(set(args.search_schedule)):
        raise ValueError("--search-schedule must contain unique, increasing step indices.")
    supported_search_steps = {10, 15, 20, 25, 30}
    if not set(args.search_schedule).issubset(supported_search_steps):
        raise ValueError(f"--search-schedule must use steps in {sorted(supported_search_steps)}.")
    if args.search_schedule[-1] >= args.sampling_steps:
        raise ValueError("Search steps must be smaller than --sampling-steps.")
    if args.samples_per_prompt < 1 or args.diversity_samples < 1:
        raise ValueError("Sample counts must be at least 1.")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    model_path = require_path(args.model_path, "Wan checkpoint")
    reward_backbone_path = require_path(args.reward_backbone_path, "Reward backbone")
    prompts = load_prompts(args.prompt, args.prompt_file, args.dimensions, args.max_prompts)
    device_id = cuda_device_index(args.device)
    seed_everything(args.seed)

    import torch

    from latsearch.reward.checkpoint import load_latent_reward_state
    from latsearch.reward.model import LatentReward
    from third_party.WanVideoModel import wan
    from third_party.WanVideoModel.wan.configs import WAN_CONFIGS

    print(f"Loading {args.model_name} from {model_path}")
    pipeline = wan.WanT2VWithLatSearch(
        config=WAN_CONFIGS[args.model_name],
        checkpoint_dir=str(model_path),
        device_id=device_id,
    )
    verifier = LatentReward(
        load_from_pretrained=str(reward_backbone_path),
        device=torch.device(args.device),
        dtype=torch.bfloat16,
    ).to(args.device)
    load_latent_reward_state(verifier, load_torch_state_dict(args.latent_reward_checkpoint))
    verifier.eval()

    for prompt_index, item in enumerate(prompts):
        repetitions = samples_for_prompt(item, args.samples_per_prompt, args.diversity_samples)
        for sample_index in range(repetitions):
            sample_seed = args.seed + sample_index
            video = pipeline.generate_with_latsearch(
                input_prompt=item.text,
                size=(args.width, args.height),
                frame_num=args.num_frames,
                shift=args.flow_shift,
                sample_solver=args.solver,
                sampling_steps=args.sampling_steps,
                guide_scale=args.guidance_scale,
                seed=sample_seed,
                offload_model=args.offload_model,
                verifier=verifier,
                search_schedule=args.search_schedule,
                num_particles=args.num_candidates,
                beta=args.beta,
                temperature=args.temperature,
            )
            path = output_path(args.output_dir, item, prompt_index, sample_seed)
            save_video(video, path, args.fps)
            print(f"Saved {path}")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
