#!/usr/bin/env bash
# Full chain on any fixed-camera clip window: detect -> SAM 2 players -> 25 fps ball -> Kalman -> teams -> passes -> video.
# usage: bash scripts/run_clip.sh <video> <start_s> <seconds> <out_dir> <band_y0> <band_y1> [pitch_mask.npy]
set -e
cd "C:/Users/Shayan Sethi/gaffer"
STRIDE=${STRIDE:-2}; BALL_STEP=${BALL_STEP:-1}; STEP_S=${STEP_S:-0.04}; export CH=${CH:-10}      # rows: every STRIDE-th video frame; ball: every BALL_STEP-th frame
V="$1"; START=$2; SECS=$3; D=$4; Y0=$5; Y1=$6; MASK=${7:-}; GPU=.local_data/venv_gpu/Scripts/python.exe
stamp() { echo "[$(date +%H:%M:%S)] $1"; }
rm -rf "$D"; mkdir -p "$D"
stamp "1/7 detect people"; Y1=$Y1 IMGSZ=1920 $GPU scripts/clip_tiles_detect.py "$V" $D $START $SECS $STRIDE $Y0 1920 240 2>&1 | grep -E "done|Traceback|Error" || true
N=$(wc -l < $D/dets.jsonl); SF=$($GPU -c "import json;m=json.load(open('$D/meta.json'));print(m['fps']/m['stride'])")
python -c "import json;m=json.load(open('$D/meta.json'));m['band_y1']=$Y1;json.dump(m,open('$D/meta.json','w'))"
stamp "   $N rows at $SF rows/s"
stamp "2/7 SAM 2 players"; PITCH_MASK=$MASK $GPU scripts/sam2_track2.py $D $Y1 70 > $D/sam2.log 2>&1 || true; grep -v -i "warn\|frame loading\|propagate in video\|Skipping\|pred_masks" $D/sam2.log | tail -20; [ -f $D/identities.json ] || { echo "FAILED: SAM 2 produced no tracks (see $D/sam2.log)"; exit 1; }
stamp "3/7 ball detection (native tiles)"; BALL_WEIGHTS=.local_data/models/football-ball-detection.pt TILE_W=1280 $GPU scripts/ball_tiles_hi.py $D $BALL_STEP $Y0 $Y1 0.08 2>&1 | grep -E "done|Traceback|Error" || true
stamp "4/7 ball track"; BK_HI=1 BK_STEP_S=$STEP_S python scripts/ball_kalman.py track $D $SF 2>&1 | tail -3
stamp "5/7 teams from bib colour"; python scripts/sam2_teams.py $D 2>&1 | head -3
stamp "6/7 passes"; python scripts/pass_events.py $D $SF 0 2>&1 | tee $D/pass_events.txt
stamp "7/7 video"; python scripts/render_sam2.py $D $N $D/passes_teams.webm 2>&1 | tail -1
stamp "ALL DONE"
