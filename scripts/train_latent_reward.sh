#!/usr/bin/env bash
set -euo pipefail

python -m tools.train_latent_reward \
    --job_id "${JOB_ID:-latsearch}" \
    --seed 3203 \
    --split_seed 1203 \
    --batch_size 4 \
    --lr 1e-4 \
    --epochs 15 \
    --milestones 10 \
    --number_frames 20 \
    --weight_VQ 1.0 \
    --weight_MQ 1.0 \
    --weight_TA 1.0 \
    --weight_CLS_VQ 1.0 \
    --weight_CLS_MQ 1.0 \
    --weight_CLS_TA 1.0 \
    --json_root_path "${METADATA_DIR:-data/metadata}" \
    --load_from_pretrained "${VIDEO_REWARD_PATH:-checkpoints/VideoReward}" \
    --output_dir "${OUTPUT_DIR:-checkpoints}" \
    --checkpoint_name "${CHECKPOINT_NAME:-latent_reward.pt}"
