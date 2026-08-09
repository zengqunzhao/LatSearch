#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-checkpoints/Wan2.1-T2V-1.3B}"
VIDEO_REWARD_PATH="${VIDEO_REWARD_PATH:-checkpoints/VideoReward}"
PROMPT="${PROMPT:-A red panda plays a tiny guitar beside a mountain stream.}"

for method in wan freeinit freqprior video-reward evosearch; do
  python generate_baseline.py \
    --method "${method}" \
    --prompt "${PROMPT}" \
    --model-path "${MODEL_PATH}" \
    --video-reward-path "${VIDEO_REWARD_PATH}" \
    --output-dir outputs/baselines
done
