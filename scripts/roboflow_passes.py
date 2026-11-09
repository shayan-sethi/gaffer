#!/usr/bin/env python3
"""
Pass counting on the cleaned tracks + ball path from roboflow_postprocess.py.

Possession: a player "has" the ball when they are the nearest player within
POSSESSION_CM of it for at least MIN_FRAMES consecutive ball frames. A pass is a
change of possession from one player to another within MAX_GAP_S, with the ball
travelling at least MIN_TRAVEL_CM between them. Same team = completed pass,
opposite team = turnover. Possession of the same player on either side of a gap
is a dribble/hold, not a pass.

Usage: python roboflow_passes.py <pipeline_dir>
"""
import sys
import os
import json

import numpy as np

D = sys.argv[1]
FPS = 25
POSSESSION_CM = 250
MIN_FRAMES = 4
MAX_GAP_S = 3.0
MIN_TRAVEL_CM = 400


def main():
    rows = {r["frame"]: r for r in json.load(open(os.path.join(D, "tracks_clean.json")))}
    ball = json.load(open(os.path.join(D, "ball_clean.json")))

    # nearest player per ball frame
    near = []
    for b in ball:
        r = rows[b["frame"]]
        cand = [(t, c, xy) for t, c, xy in r["players"] + r["goalkeepers"]]
        if not cand:
            near.append(None)
            continue
        d = [np.hypot(xy[0] - b["pitch_cm"][0], xy[1] - b["pitch_cm"][1]) for _, _, xy in cand]
        k = int(np.argmin(d))
        near.append((cand[k][0], cand[k][1], d[k]) if d[k] <= POSSESSION_CM else None)

    # runs of the same nearest player (consecutive ball frames, no frame gap > 3)
    runs = []
    for b, n in zip(ball, near):
        if n is None:
            continue
        if runs and runs[-1]["player"] == n[0] and b["frame"] - runs[-1]["end"] <= 3:
            runs[-1]["end"] = b["frame"]
            runs[-1]["n"] += 1
            runs[-1]["end_pos"] = b["pitch_cm"]
        else:
            runs.append({"player": n[0], "team": n[1], "start": b["frame"], "end": b["frame"], "n": 1,
                         "start_pos": b["pitch_cm"], "end_pos": b["pitch_cm"]})
    runs = [r for r in runs if r["n"] >= MIN_FRAMES]
    # merge same-player runs separated by a short gap (a dribble / brief loss)
    merged = []
    for r in runs:
        if merged and merged[-1]["player"] == r["player"] and r["start"] - merged[-1]["end"] <= MAX_GAP_S * FPS:
            merged[-1]["end"], merged[-1]["end_pos"] = r["end"], r["end_pos"]
            merged[-1]["n"] += r["n"]
        else:
            merged.append(dict(r))

    events = []
    for a, b in zip(merged, merged[1:]):
        gap = (b["start"] - a["end"]) / FPS
        travel = float(np.hypot(b["start_pos"][0] - a["end_pos"][0], b["start_pos"][1] - a["end_pos"][1]))
        if gap > MAX_GAP_S or travel < MIN_TRAVEL_CM:
            continue
        events.append({"from": a["player"], "to": b["player"], "team_from": a["team"], "team_to": b["team"],
                       "kind": "pass" if a["team"] == b["team"] else "turnover",
                       "frame_release": a["end"], "frame_receive": b["start"],
                       "travel_m": round(travel / 100, 1), "gap_s": round(gap, 2)})
    json.dump({"possession_runs": merged, "events": events}, open(os.path.join(D, "passes.json"), "w"), indent=1)
    p = [e for e in events if e["kind"] == "pass"]
    print("possession runs:", len(merged))
    for e in events:
        print(e)
    for t in (0, 1):
        print(f"team {t}: passes {sum(1 for e in p if e['team_from'] == t)}, turnovers lost {sum(1 for e in events if e['kind']=='turnover' and e['team_from']==t)}")


if __name__ == "__main__":
    main()
