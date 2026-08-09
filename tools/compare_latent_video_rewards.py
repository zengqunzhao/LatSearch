import argparse
import json
import os
import random

import numpy as np
import torch

from latsearch.cli_utils import load_torch_state_dict
from latsearch.reward.checkpoint import load_latent_reward_state
from latsearch.reward.model import LatentReward
from third_party.VideoReward.score import VideoVLMRewardInference
from third_party.WanVideoModel import wan
from third_party.WanVideoModel.wan.configs import WAN_CONFIGS

parser = argparse.ArgumentParser(description="VideoGen")
parser.add_argument("--infer_step", type=int, default=50, help="total inference timestep T")
parser.add_argument("--frame_number", type=int, default=33, help="number of video frames")
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed to determine the initial latent."
)
parser.add_argument(
    "--device", type=str, default="cuda", help="Device where the model inference is performed."
)
parser.add_argument("--model_name", type=str, default="t2v-1.3B", help="pre-trained model name")
parser.add_argument(
    "--model_path", type=str, default="./checkpoints/Wan2.1-T2V-1.3B", help="pre-trained model path"
)
parser.add_argument("--reward_model_path", type=str, required=True)
parser.add_argument(
    "--prompt_path", type=str, default="./prompts/VBench2_full_info.json", help="prompt file path"
)
parser.add_argument(
    "--dimension_list",
    nargs="+",
    default=["Camera_Motion"],
    help="List of dimensions (one or more)",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="./outputs/reward_comparison",
    help="Path to save generated videos.",
)
parser.add_argument("--log_dir", type=str, default="./logs/reward_comparison")
parser.add_argument("--rewards_type", type=str, default="latent_vs_decoded")
parser.add_argument("--video_reward_path", type=str, default="./checkpoints/VideoReward")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    dtype = torch.float16
    device = torch.device(args.device)
    cfg = WAN_CONFIGS[args.model_name]

    pipe = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.model_path,
        device_id=0,
    )

    ######## Video Reward ########
    load_from_pretrained = args.video_reward_path
    verifier_video = VideoVLMRewardInference(load_from_pretrained, device=device, dtype=dtype)
    ######## Latent Reward ########
    verifier_latent = LatentReward(
        load_from_pretrained=args.video_reward_path,
        device=device,
        dtype=torch.bfloat16,
    ).to(device)
    load_latent_reward_state(verifier_latent, load_torch_state_dict(args.reward_model_path))
    verifier_latent.eval()

    # Load prompts
    with open(args.prompt_path) as f:
        prompts = json.load(f)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    for idx, prompt_ in enumerate(prompts):
        dimension = prompt_["dimension"][0]
        if dimension not in args.dimension_list:
            continue
        log_path = os.path.join(args.log_dir, f"{args.rewards_type}_{dimension}.jsonl")

        num_iters = 20 if dimension == "Diversity" else 3
        prompt = prompt_["prompt_en"]

        # folder for this dimension
        dim_save_dir = os.path.join(args.save_dir, dimension)
        os.makedirs(dim_save_dir, exist_ok=True)

        for run_idx in range(num_iters):
            cur_seed = args.seed + run_idx

            # === Generate + get intermediate rewards ===
            video, latent_reward_dict, video_reward_dict = pipe.generate_get_reward(
                input_prompt=prompt,
                size=(832, 480),
                frame_num=args.frame_number,
                shift=5.0,
                sample_solver="unipc",
                sampling_steps=args.infer_step,
                guide_scale=5.0,
                seed=cur_seed,
                verifier_video=verifier_video,
                verifier_latent=verifier_latent,
            )

            # Compute final video reward (Overall)
            if video is not None:
                video_batch = video.unsqueeze(0)
                rewards = verifier_video.reward(
                    video_batch.permute(0, 2, 1, 3, 4), [prompt], use_norm=False
                )
                final_overall = float(rewards[0]["Overall"])
            else:
                final_overall = None

            # === Format one result entry ===
            result_entry = {
                "prompt_idx": idx,
                "prompt": prompt,
                "dimension": dimension,
                "seed": cur_seed,
                "latent_rewards": latent_reward_dict,  # {step: value}
                "decoded_rewards": video_reward_dict,  # {step: value}
                "final_video_reward": final_overall,
            }

            # === Incrementally write to JSONL ===
            with open(log_path, "a") as f:
                f.write(json.dumps(result_entry) + "\n")

            print(f"[SAVED] step-wise rewards to {log_path}")
