#!/usr/bin/env bash
# One-command setup for a rented Linux GPU box (RunPod / Vast.ai / AWS). Use a PyTorch template (Python 3.10+, CUDA) so torch is already installed.
#   git clone --branch experiments/pitch-team-identity-research https://github.com/shayan-sethi/gaffer.git && cd gaffer && bash scripts/cloud_setup.sh
# Then upload (scp / the provider's file browser) into the gaffer folder:
#   match.mp4, best.pt -> .local_data/ball_scale_model/run/weights/best.pt, pitch_mask.npy -> .local_data/pitch_mask.npy
set -e
cd "$(dirname "$0")/.."
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "WARNING: no NVIDIA GPU visible"
python3 --version
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || pip install torch torchvision
apt-get install -y ffmpeg >/dev/null 2>&1 || echo "(ffmpeg not installed via apt; install it if rendering fails)"
pip install -q -r scripts/requirements-pipeline.txt
SAM2_BUILD_CUDA=0 pip install -q git+https://github.com/facebookresearch/sam2.git
bash scripts/get_weights.sh
mkdir -p .local_data/ball_scale_model/run/weights
cat <<'EOF'

Setup done. Upload best.pt, pitch_mask.npy and the video, then time one clip:

  time PY=python3 bash scripts/run_clip.sh /path/match.mp4 2594 30 .local_data/cloud_out 560 1440 .local_data/pitch_mask.npy

Compare the 'real' time with the laptop (about 16 min for 30 s at full rate).
STOP / DELETE THE INSTANCE when finished, or it keeps billing.
EOF
