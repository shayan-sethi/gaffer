#!/usr/bin/env bash
# Downloads the model weights that can be fetched publicly. Run once after cloning:  bash scripts/get_weights.sh
# (The trained ball model and the pitch masks are NOT in git: copy them from the machine they were made on, see docs/SETUP_MAC.md.)
set -e
cd "$(dirname "$0")/.."
mkdir -p .local_data/ext/sam2_ckpt .local_data/models
# SAM 2.1 small (175 MB) from the public Hugging Face copy (no login needed)
if [ ! -f .local_data/ext/sam2_ckpt/sam2.1_hiera_small.pt ]; then
  curl -L --fail -o .local_data/ext/sam2_ckpt/sam2.1_hiera_small.pt https://huggingface.co/facebook/sam2.1-hiera-small/resolve/main/sam2.1_hiera_small.pt
fi
ls -la .local_data/ext/sam2_ckpt
echo "yolov8n.pt (person detector) is downloaded automatically by ultralytics on first use."
echo "Still to copy by hand: .local_data/ball_scale_model/run/weights/best.pt (trained ball model) and any pitch mask (.npy)."
