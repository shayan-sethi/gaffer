#!/usr/bin/env bash
# Full pipeline on a longer stretch of the fog clip (6v6, pink bibs vs no bibs). One chain, logs each stage with a timestamp.
# usage: bash scripts/run_fog90.sh <start_s> <seconds> <out_dir>
set -e
cd "$(dirname "$0")/.."
START=${1:-35}; SECS=${2:-90}; D=${3:-.local_data/fog_90s}
V="test_videos/Saturday Morning Football in Heavy Fog 10am Match Highlights 7th November 2026 Eltham Goals.mp4"
GPU=.local_data/venv_gpu/Scripts/python.exe
export PITCH_MASK=.local_data/fog_pitch_mask.npy COLOUR_LOCK=1 EXCLUDE_EDGE_PX=4 SAT_W=120 TEAM_LABELS="W:P,B:B" LBL_W=P LBL_B=B TEAM_SIZE=6 RETEAM=1
stamp() { echo "[$(date +%H:%M:%S)] $1"; }
stamp "1/8 detect people (GPU)"
Y1=1080 IMGSZ=2560 $GPU scripts/clip_tiles_detect.py "$V" $D $START $SECS 5 380 1920 240 2>&1 | grep -E "done|Traceback|Error" || true
stamp "   rows: $(wc -l < $D/dets.jsonl)"
stamp "2/8 shirt colours"
python scripts/clip_colours.py $D 2>&1 | grep -E "done|Traceback|Error" || true
stamp "3/8 identities (12 players)"
python scripts/dump_identities.py "$V" $D $START $SECS 12 2>&1 | grep -E "bib colour|bib share|^identities|Traceback|Error" || true
stamp "4/8 ball detection (GPU, trained white-ball model)"
BALL_WEIGHTS=.local_data/fog_ball/ball_ft2/weights/best.pt $GPU scripts/clip_ball_tiles.py $D 2 380 1080 0.08 2>&1 | grep -E "done|Traceback|Error" || true
stamp "5/8 Kalman ball track"
python scripts/ball_kalman.py track $D 10 2>&1 | tail -3
stamp "6/8 identities with labels, then touch-based passes"
BALL_TRACK=$D/ball_track.csv FAST=1 python scripts/clip_passes.py "$V" $D $START $SECS 12 2>&1 | grep -E "identities by team|Traceback|Error" || true
python scripts/pass_events.py $D 10 0 2>&1 | tee $D/pass_events.txt | head -40
stamp "7/8 report, plan view"
python scripts/match_report.py $D "$V" $START $SECS 10 2>&1 | grep -v -i warn | tail -6
python scripts/plan_view.py $D 110 "140,548,300,520" "1838,522,1700,470" 2>&1 | grep -E "camera|Traceback|Error" || true
stamp "8/8 render video"
BALL_TRACK=$D/ball_track.csv PASS_EVENTS=$D/pass_events.json VIEW="0,1920,380,1080,0.7" python scripts/clip_passes.py "$V" $D $START $SECS 12 2>&1 | grep -E "wrote|Traceback|Error" || true
stamp "ALL DONE"
