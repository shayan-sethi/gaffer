# Running the video pipeline on a Mac (or any machine)

The pipeline in `scripts/` runs on Windows + NVIDIA, on a Mac (Apple GPU through PyTorch MPS) and on a plain CPU.
`scripts/device.py` picks CUDA, then MPS, then the CPU; override with `DEVICE=cpu`.

## 1. Get the code

```bash
git clone https://github.com/shayan-sethi/gaffer.git && cd gaffer
git checkout experiments/pitch-team-identity-research
```

## 2. Python environment (Python 3.10 or newer)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision
pip install -r scripts/requirements-pipeline.txt
pip install git+https://github.com/facebookresearch/sam2.git
```

If the SAM 2 install tries to build a CUDA extension and fails, retry with `SAM2_BUILD_CUDA=0 pip install git+https://github.com/facebookresearch/sam2.git`.

## 3. Weights

```bash
bash scripts/get_weights.sh
```

This downloads SAM 2.1 small (175 MB, public Hugging Face copy). The person detector (`yolov8n.pt`) downloads itself on first use.

These are NOT in git. Copy them from the machine they were made on (AirDrop, a shared drive):

| File | Where it goes |
|---|---|
| trained ball model `best.pt` (a few MB) | `.local_data/ball_scale_model/run/weights/best.pt` |
| pitch mask for a camera, e.g. `aug_pitch_mask.npy` | anywhere, passed as the last argument |
| the match video | anywhere (never in git) |

Without the trained ball model the pipeline falls back to `.local_data/models/football-ball-detection.pt` (the stock model, 68 M parameters,
much slower and weaker); copy that too if you want the fallback.

## 4. Run a clip

```bash
bash scripts/run_clip.sh "/path/to/match.mp4" 2891 30 .local_data/out_test 560 1000 /path/to/pitch_mask.npy
```

Arguments: video, start second, seconds, output folder, top and bottom of the pitch band in pixels, optional pitch mask. Output: `passes_teams.mp4`
(`.webm` on Windows), `pass_events.txt`, `teams_report.txt`, `sam2_track_report.txt`.

Speed knobs: `STRIDE=4 BALL_STEP=2 STEP_S=0.08 CH=5` tracks at half the frame rate (about twice as fast, slightly less accurate).

## What to expect on a Mac (untested)

This was developed on Windows with a 4 GB NVIDIA laptop GPU. The Mac path has not been run yet. Things to watch for:
- MPS is missing a few operations; `PYTORCH_ENABLE_MPS_FALLBACK=1` (set by the script) sends those to the CPU.
- bfloat16 autocast is only used on CUDA; on MPS SAM 2 runs in full precision (more memory, maybe slower per frame).
- Time the same 20-30 s clip on both machines before planning a full match.
