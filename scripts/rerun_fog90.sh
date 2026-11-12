#!/usr/bin/env bash
# Re-run everything after detection (cached): identities -> passes -> report -> video.   bash scripts/rerun_fog90.sh <start_s> <seconds> <out_dir>
set -e
cd "$(dirname "$0")/.."
START=${1:-35}; SECS=${2:-90}; D=${3:-.local_data/fog_90s}
V="test_videos/Saturday Morning Football in Heavy Fog 10am Match Highlights 7th November 2026 Eltham Goals.mp4"
export PITCH_MASK=.local_data/fog_pitch_mask.npy COLOUR_LOCK=1 EXCLUDE_EDGE_PX=4 SAT_W=120 TEAM_LABELS="W:P,B:B" LBL_W=P LBL_B=B TEAM_SIZE=6 RETEAM=1
stamp() { echo "[$(date +%H:%M:%S)] $1"; }
stamp "identities"; python scripts/dump_identities.py "$V" $D $START $SECS 12 2>&1 | grep -E "bib share|^identities|Traceback|Error" || true
stamp "identities with labels"; BALL_TRACK=$D/ball_track.csv FAST=1 python scripts/clip_passes.py "$V" $D $START $SECS 12 2>&1 | grep -E "identities by team|Traceback|Error" || true
stamp "passes"; python scripts/pass_events.py $D 10 0 2>&1 | tee $D/pass_events.txt | head -40
stamp "report"; python scripts/match_report.py $D "$V" $START $SECS 10 2>&1 | grep -E "wrote|Traceback|Error" || true
python scripts/plan_view.py $D 110 "140,548,300,520" "1838,522,1700,470" 2>&1 | grep -E "Traceback|Error" || true
stamp "video"; BALL_TRACK=$D/ball_track.csv PASS_EVENTS=$D/pass_events.json VIEW="0,1920,380,1080,0.7" python scripts/clip_passes.py "$V" $D $START $SECS 12 2>&1 | grep -E "wrote|Traceback|Error" || true
stamp "DONE"
