#!/usr/bin/env bash
# Full chain on any fixed-camera clip window: detect -> SAM 2 players -> 25 fps ball -> Kalman -> teams -> passes -> video.
# Runs on Windows (Git Bash) with an NVIDIA GPU, on a Mac (Apple GPU through MPS) and on a plain CPU. Run it from anywhere.
# usage: bash scripts/run_clip.sh <video> <start_s> <seconds> <out_dir> <band_y0> <band_y1> [pitch_mask.npy]
# env:   STRIDE (2 = 12.5 rows/s on a 25 fps clip), BALL_STEP (1 = every frame), STEP_S (ball step in seconds), CH (SAM 2 correction chunk, rows),
#        BALL_WEIGHTS (default: the zoom-trained ball model if present, else the stock one), PY (python to use), DEVICE (cpu | cuda | mps)
set -e
cd "$(dirname "$0")/.."
if [ -n "$PY" ]; then :; elif [ -x .local_data/venv_gpu/Scripts/python.exe ]; then PY=.local_data/venv_gpu/Scripts/python.exe; elif command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
export PYTORCH_ENABLE_MPS_FALLBACK=1
STRIDE=${STRIDE:-2}; BALL_STEP=${BALL_STEP:-1}; STEP_S=${STEP_S:-0.04}; export CH=${CH:-10}
V="$1"; START=$2; SECS=$3; D=$4; Y0=$5; Y1=$6; MASK=${7:-}
if [ -z "$BALL_WEIGHTS" ]; then
  if [ -f .local_data/ball_scale_model/run/weights/best.pt ]; then BALL_WEIGHTS=.local_data/ball_scale_model/run/weights/best.pt; else BALL_WEIGHTS=.local_data/models/football-ball-detection.pt; fi
fi
stamp() { echo "[$(date +%H:%M:%S)] $1"; }
rm -rf "$D"; mkdir -p "$D"
stamp "1/7 detect people"; Y1=$Y1 IMGSZ=1920 $PY scripts/clip_tiles_detect.py "$V" $D $START $SECS $STRIDE $Y0 1920 240 2>&1 | grep -E "done|Traceback|Error" || true
[ -s $D/dets.jsonl ] || { echo "FAILED: no detections written"; exit 1; }
N=$(wc -l < $D/dets.jsonl | tr -d ' '); SF=$($PY -c "import json;m=json.load(open('$D/meta.json'));print(m['fps']/m['stride'])")
$PY -c "import json;m=json.load(open('$D/meta.json'));m['band_y1']=$Y1;json.dump(m,open('$D/meta.json','w'))"
stamp "   $N rows at $SF rows/s"
stamp "2/7 SAM 2 players"; PITCH_MASK=$MASK $PY scripts/sam2_track2.py $D $Y1 70 > $D/sam2.log 2>&1 || true
grep -v -i "warn\|frame loading\|propagate in video\|Skipping\|pred_masks" $D/sam2.log | tail -20
[ -f $D/identities.json ] || { echo "FAILED: SAM 2 produced no tracks (see $D/sam2.log)"; exit 1; }
stamp "3/7 ball detection (native tiles, weights $BALL_WEIGHTS)"; BALL_WEIGHTS=$BALL_WEIGHTS TILE_W=1280 $PY scripts/ball_tiles_hi.py $D $BALL_STEP $Y0 $Y1 0.08 2>&1 | grep -E "done|Traceback|Error" || true
[ -s $D/ball_hi.jsonl ] || { echo "FAILED: no ball detections written"; exit 1; }
stamp "4/7 ball track"; BK_HI=1 BK_STEP_S=$STEP_S $PY scripts/ball_kalman.py track $D $SF 2>&1 | tail -3
stamp "5/7 teams from bib colour"; $PY scripts/sam2_teams.py $D 2>&1 | head -3
stamp "6/7 passes"; $PY scripts/pass_events.py $D $SF 0 2>&1 | tee $D/pass_events.txt
stamp "7/7 video"; $PY scripts/render_sam2.py $D $N $D/passes_teams.webm 2>&1 | grep -E "wrote|Traceback|Error" || true
stamp "ALL DONE"
