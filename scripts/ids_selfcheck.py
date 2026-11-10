#!/usr/bin/env python3
"""Self-check for an identity lock: python ids_selfcheck.py <video> <out_dir> <start_s> <seconds> <n> [greedy]"""
import sys, json, numpy as np
video, out, st, sec, n = sys.argv[1:6]
greedy = len(sys.argv) > 6
sys.argv = ["clip_ids.py", video, out, st, sec, n]
sys.path.insert(0, "scripts")
import clip_ids as C
meta = json.load(open(out + "/meta.json"))
rows = sorted((json.loads(l) for l in open(out + "/dets.jsonl")), key=lambda r: r["frame"])
sfps = meta["fps"] / meta["stride"]
pf, _ = C.track(rows, np.ones((meta["height"], meta["width"]), np.uint8), [], sfps)
ident, un, nraw, nlong, _ = (C.lock_identities_greedy if greedy else C.lock_identities)(pf, sfps)
print(f"raw ids {nraw}, tracklets {nlong}, identities {len(ident)}")
tot = 0
H = None
for k, I in enumerate(ident):
    fr = sorted(f for f, o in I["frames"].items() if not o.get("interp"))
    b = np.array([np.mean(I["frames"][f]["rgb"]) for f in fr])
    m = np.array([np.median(b[max(0, i - 7):i + 8]) for i in range(len(b))])
    fl = int(((m > 120).astype(int)[1:] != (m > 120).astype(int)[:-1]).sum()); tot += fl
    ft = np.array([C.foot(I["frames"][f]) for f in fr]); h = np.median([C.height(I["frames"][f]) for f in fr])
    step = np.hypot(*np.diff(ft, axis=0).T); jumps = int((step > 2.5 * h).sum())      # > 2.5 body heights between samples
    print(f"  {I['team']}{k} {I['role']}{'?' if I.get('guess') else ''} out={I.get('outlier',0):.1f} frames {len(I['frames']):4d} (real {len(fr):4d}) tracklets {len(I['tids']):2d} colour-flips {fl} jumps {jumps} extent {np.ptp(ft[:,0]):.0f}x{np.ptp(ft[:,1]):.0f}px")
cnt = [sum(1 for I in ident if fi in I['frames']) for fi in range(len(rows))]
big = sorted([len(f) for f, _ in un], reverse=True)
print(f"total colour flips {tot} | mean identified {np.mean(cnt):.2f}/{n} | all {n}: {100*np.mean([c==int(n) for c in cnt]):.0f}% | >=8: {100*np.mean([c>=8 for c in cnt]):.0f}% | unassigned frames {sum(big)} biggest {big[:6]}")
