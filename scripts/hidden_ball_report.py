#!/usr/bin/env python3
"""For every stretch where the ball is hidden: who had it last, who has it next, and what the players do in between.
python hidden_ball_report.py <out_dir> [min_seconds=1.0]"""
import sys, json, numpy as np
D = sys.argv[1]; MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
j = json.load(open(D + "/passes.json")); ids = json.load(open(D + "/identities.json"))
ball = {int(k): v for k, v in json.load(open(D + "/ball_path.json")).items()}; est = {int(k): v for k, v in json.load(open(D + "/ball_est.json")).items()}
lab = {I["label"]: I for I in ids.values()}
seen = sorted(ball); sfps = 10.0
def holders(r):
    out = []
    for I in ids.values():
        b = I["rows"].get(str(r))
        if b and I["role"] != "REF":
            h = b[3] - b[1]; p = ball.get(r) or est.get(r)
            if p: out.append((float(np.hypot(p[0] - (b[0] + b[2]) / 2, p[1] - (b[3] - 0.1 * h)) / h), I["label"]))
    return sorted(out)
for a, b in zip(seen, seen[1:]):
    if (b - a) / sfps < MIN: continue
    ha, hb = holders(a), holders(b)
    print(f"hidden {a/sfps:5.1f}s -> {b/sfps:5.1f}s ({(b-a)/sfps:.1f} s)")
    print(f"   last seen at {a/sfps:.1f}s, nearest players: {[(n, round(d,1)) for d, n in ha[:3]]}")
    print(f"   next seen at {b/sfps:.1f}s, nearest players: {[(n, round(d,1)) for d, n in hb[:3]]}")
    # which players are inside the zone where the ball could be: within 6 body-heights of the straight line between the two sightings
    pa, pb = np.array(ball[a]), np.array(ball[b]); cand = {}
    for r in range(a, b + 1, 5):
        for I in ids.values():
            bx = I["rows"].get(str(r))
            if bx and I["role"] != "REF":
                h = bx[3] - bx[1]; f = np.array([(bx[0] + bx[2]) / 2, bx[3]]); t = (r - a) / (b - a); line = pa * (1 - t) + pb * t
                cand.setdefault(I["label"], []).append(float(np.linalg.norm(f - line) / h))
    close = sorted(((np.min(v), k) for k, v in cand.items()))[:5]
    print("   players who came closest to the ball's likely path:", [(k, round(float(d), 1)) for d, k in close])
    team_a = ha[0][1][0] if ha else "?"; team_b = hb[0][1][0] if hb else "?"
    print("   verdict:", "same player has it before and after" if ha and hb and ha[0][1] == hb[0][1] else (f"possession probably changed hands ({ha[0][1] if ha else '?'} -> {hb[0][1] if hb else '?'}), team {team_a} -> {team_b}, path unseen: NOT counted as a pass"))
