#!/usr/bin/env python3
"""Generate videos with the baselines evaluated in the LatSearch paper."""

from __future__ import annotations

import argparse

from latsearch.cli_utils import (
    cuda_device_index,
    load_prompts,
    output_path,
    require_path,
    samples_for_prompt,
    save_video,
    seed_everything,
)

METHODS = ("wan", "freeinit", "freqprior", "video-reward", "evosearch")


def build_parser(default_method: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a LatSearch evaluation baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        required=default_method is None,
        default=default_method,
    )
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file")
    parser.add_argument("--dimensions", nargs="+")
    parser.add_argument("--max-prompts", type=int)

    parser.add_argument("--model-name", choices=("t2v-1.3B", "t2v-14B"), default="t2v-1.3B")
    parser.add_argument("--model-path", default="checkpoints/Wan2.1-T2V-1.3B")
    parser.add_argument("--video-reward-path", default="checkpoints/VideoReward")
    parser.add_argument("--output-dir", default="outputs/baselines")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--diversity-samples", type=int, default=20)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--offload-model", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--num-candidates", type=int, default=4, help="VideoReward best-of-N budget."
    )
    parser.add_argument(
        "--freeinit-iters", type=int, default=5, help="Initial pass plus four refinements."
    )
    parser.add_argument(
        "--freqprior-iters", type=int, default=3, help="Initial pass plus two refinements."
    )
    parser.add_argument("--freqprior-ratio", type=float, default=0.8)
    parser.add_argument("--evolution-schedule", nargs="+", type=int, default=[5, 20])
    parser.add_argument("--population-schedule", nargs="+", type=int, default=[6, 3, 3])
    parser.add_argument("--elite-size", type=int, default=3)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_frames < 1 or (args.num_frames - 1) % 4:
        raise ValueError("--num-frames must have the form 4n+1.")
    if args.samples_per_prompt < 1 or args.diversity_samples < 1:
        raise ValueError("Sample counts must be at least 1.")
    if args.method == "evosearch":
        if len(args.population_schedule) != len(args.evolution_schedule) + 1:
            raise ValueError(
                "The population schedule needs one more value than the evolution schedule."
            )
        if args.elite_size > min(args.population_schedule):
            raise ValueError("--elite-size cannot exceed the smallest population size.")


def _build_pipeline(
    args: argparse.Namespace, wan: object, config: object, device_id: int
) -> object:
    classes = {
        "wan": wan.WanT2V,
        "freeinit": wan.WanT2VWithFreeInit,
        "freqprior": wan.WanT2VWithFreqPrior,
        "video-reward": wan.WanT2V,
        "evosearch": wan.WanT2VWithEvoSearch,
    }
    return classes[args.method](config=config, checkpoint_dir=args.model_path, device_id=device_id)


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    args.model_path = str(require_path(args.model_path, "Wan checkpoint"))
    prompts = load_prompts(args.prompt, args.prompt_file, args.dimensions, args.max_prompts)
    device_id = cuda_device_index(args.device)
    seed_everything(args.seed)

    import torch

    from third_party.WanVideoModel import wan
    from third_party.WanVideoModel.wan.configs import WAN_CONFIGS

    verifier = None
    if args.method in {"video-reward", "evosearch"}:
        from third_party.VideoReward.score import VideoVLMRewardInference

        reward_path = require_path(args.video_reward_path, "VideoReward checkpoint")
        verifier = VideoVLMRewardInference(
            str(reward_path), device=torch.device(args.device), dtype=torch.float16
        )

    pipeline = _build_pipeline(args, wan, WAN_CONFIGS[args.model_name], device_id)
    for prompt_index, item in enumerate(prompts):
        repetitions = samples_for_prompt(item, args.samples_per_prompt, args.diversity_samples)
        for sample_index in range(repetitions):
            sample_seed = args.seed + sample_index
            common = dict(
                input_prompt=item.text,
                size=(args.width, args.height),
                frame_num=args.num_frames,
                shift=args.flow_shift,
                sampling_steps=args.sampling_steps,
                guide_scale=args.guidance_scale,
                seed=sample_seed,
                offload_model=args.offload_model,
            )
            if args.method == "wan":
                video = pipeline.generate(sample_solver="unipc", **common)
            elif args.method == "freeinit":
                video = pipeline.generate(
                    sample_solver="unipc", num_iters=args.freeinit_iters, **common
                )
            elif args.method == "freqprior":
                video = pipeline.generate(
                    sample_solver="unipc",
                    num_iters=args.freqprior_iters,
                    ratio=args.freqprior_ratio,
                    **common,
                )
            elif args.method == "video-reward":
                timing = {"DiT_Time": [], "Decoder_Time": []}
                candidates = pipeline.generate_N_videos(
                    sample_solver="unipc",
                    number_of_N=args.num_candidates,
                    time_dic=timing,
                    **common,
                )
                candidates = torch.stack(candidates)
                scores = verifier.reward(
                    candidates.permute(0, 2, 1, 3, 4),
                    [item.text] * len(candidates),
                )
                best = max(range(len(scores)), key=lambda index: scores[index]["Overall"])
                video = candidates[best]
            else:
                generator = torch.Generator(device="cpu").manual_seed(sample_seed)
                target_shape = (
                    pipeline.vae.model.z_dim,
                    (args.num_frames - 1) // pipeline.vae_stride[0] + 1,
                    args.height // pipeline.vae_stride[1],
                    args.width // pipeline.vae_stride[2],
                )
                population = torch.randn(
                    (args.population_schedule[0], *target_shape),
                    generator=generator,
                    dtype=torch.float16,
                ).to(args.device)
                timing = {"DiT_Time": [], "Decoder_Time": [], "Reward_Time": []}
                video = pipeline.generate(
                    input_prompt=item.text,
                    size=(args.width, args.height),
                    frame_num=args.num_frames,
                    shift=args.flow_shift,
                    guide_scale=args.guidance_scale,
                    sampling_steps=args.sampling_steps,
                    noise=population,
                    seed=sample_seed,
                    verifier=verifier,
                    number_of_N=len(population),
                    sample_solver="dpm++",
                    evolution_schedule=args.evolution_schedule,
                    population_size_schedule=args.population_schedule,
                    mutation_rate=args.mutation_rate,
                    elite_size=args.elite_size,
                    generator=generator,
                    offload_model=args.offload_model,
                    time_dic=timing,
                )[0]

            path = output_path(f"{args.output_dir}/{args.method}", item, prompt_index, sample_seed)
            save_video(video, path, args.fps)
            print(f"Saved {path}")


def main(default_method: str | None = None) -> None:
    run(build_parser(default_method).parse_args())


if __name__ == "__main__":
    main()
