#!/usr/bin/env python3
"""How many on-pitch players does each detection pass find per frame?  python zoom_compare.py <out_dir> <n_players>
env: PITCH_MASK, EXCLUDE_MIN_H, EXCLUDE_BOTTOM_Y"""
import sys, os, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from pitch_mask import feet_on_pitch
D, N = sys.argv[1], int(sys.argv[2])
meta = json.load(open(D + "/meta.json"))
mask = np.load(os.environ["PITCH_MASK"]).astype(np.uint8) if os.environ.get("PITCH_MASK") else np.ones((meta["height"], meta["width"]), np.uint8)
MINH, BOT = float(os.environ.get("EXCLUDE_MIN_H", 1e9)), float(os.environ.get("EXCLUDE_BOTTOM_Y", 1e9))
def count(d): return sum(1 for x in d if x[4] >= 0.25 and 18 <= x[3] - x[1] < MINH and x[3] < BOT and feet_on_pitch(mask, x[:4]))
out = {}
for name in ("dets_pass1.jsonl", "dets_nolab.jsonl"):
    c = np.array([count(json.loads(l)["dets"]) for l in open(D + "/" + name)])
    out[name] = c
    print(f"{name:18s} people on the pitch per frame: mean {c.mean():.2f} | exactly {N}: {100*np.mean(c==N):.0f}% | fewer than {N}: {100*np.mean(c<N):.0f}% | more than {N}: {100*np.mean(c>N):.0f}%")
a, b = out["dets_pass1.jsonl"], out["dets_nolab.jsonl"]
m = min(len(a), len(b))
print("frames where the zoom pass found more people:", int(np.sum(b[:m] > a[:m])), "| fewer:", int(np.sum(b[:m] < a[:m])), "| of", m)
np.save(D + "/zoom_gain.npy", (b[:m] - a[:m]))
