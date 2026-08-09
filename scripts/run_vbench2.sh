#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-checkpoints/Wan2.1-T2V-1.3B}"
VIDEO_REWARD_PATH="${VIDEO_REWARD_PATH:-checkpoints/VideoReward}"
LATENT_REWARD_PATH="${LATENT_REWARD_PATH:-checkpoints/latent_reward.pt}"

python generate.py \
  --prompt-file prompts/VBench2_full_info.json \
  --model-path "${MODEL_PATH}" \
  --reward-backbone-path "${VIDEO_REWARD_PATH}" \
  --latent-reward-checkpoint "${LATENT_REWARD_PATH}" \
  --samples-per-prompt 3 \
  --diversity-samples 20 \
  --output-dir outputs/vbench2/latsearch
