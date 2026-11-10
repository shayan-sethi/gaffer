#!/usr/bin/env python3
"""Does every identity keep ONE team colour? Uses the cached grass-free colours (run clip_colours.py first).
python team_purity.py <video> <out_dir> <start_s> <seconds> <n>        env COLOUR_LOCK=1 chains tracklets with the grass-free colour"""
import sys, os, json, numpy as np
video, out, st, sec, n = sys.argv[1:6]
sys.argv = ["clip_ids.py", video, out, st, sec, n]
sys.path.insert(0, "scripts")
import clip_ids as C
meta = json.load(open(out + "/meta.json"))
rows = sorted((json.loads(l) for l in open(out + "/dets.jsonl")), key=lambda r: r["frame"]); sfps = meta["fps"] / meta["stride"]
pf, _ = C.track(rows, np.ones((meta["height"], meta["width"]), np.uint8), [], sfps)
ident, un, *_ = C.lock_identities(pf, sfps, colourer=C.TrackletColour() if os.environ.get("COLOUR_LOCK") else None)
C.reteam(video, rows, ident)
centres = {}
for tm in ("W", "B"):
    ms = [np.median([o["lab"] for o in I["frames"].values() if o.get("lab") is not None], axis=0) for I in ident
          if I["team"] == tm and sum(1 for o in I["frames"].values() if o.get("lab") is not None) >= 3]
    centres[tm] = np.median(ms, axis=0)
bad = tot = 0
for k, I in enumerate(ident):
    if I["team"] not in ("W", "B"): continue
    tot += 1
    votes = []
    for o in I["frames"].values():
        if o.get("lab") is None: continue
        v = np.array(o["lab"]); votes.append("W" if np.linalg.norm(v - centres["W"]) < np.linalg.norm(v - centres["B"]) else "B")
    if len(votes) >= 5:
        minority = min(votes.count("W"), votes.count("B")) / len(votes)
        if minority > 0.06:
            bad += 1; print(f"  MIXED identity {k} labelled {I['team']}: {100*minority:.0f}% of its frames look like the other team  (tracklets {I['tids'][:5]})")
print(f"mixed identities: {bad} of {tot}")
