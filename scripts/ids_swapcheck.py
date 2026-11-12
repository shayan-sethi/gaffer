#!/usr/bin/env python3
"""Identity-swap check without labels. Finds every moment two same-team identities come within 0.6 body-heights, then asks whether the
labels CONTINUE each player's own motion across the crossing (before -> after) better than if the labels were swapped.
'suspect' = swapping the labels would fit the motion better than keeping them (a likely identity swap).
python ids_swapcheck.py <video> <out_dir> <start_s> <seconds> <n>     env: PRED_W, PITCH_MASK, EXCLUDE_LAB, COLOUR_LOCK"""
import sys, os, json, numpy as np
video, out, st, sec, n = sys.argv[1:6]
sys.argv = ["clip_ids.py", video, out, st, sec, n]
sys.path.insert(0, "scripts")
import clip_ids as C
meta = json.load(open(out + "/meta.json"))
rows = sorted((json.loads(l) for l in open(out + "/dets.jsonl")), key=lambda r: r["frame"]); sfps = meta["fps"] / meta["stride"]
pf, _ = C.track(rows, C.load_mask(meta), [], sfps)
ident, un, *_ = C.lock_identities(pf, sfps, colourer=C.TrackletColour() if os.environ.get("COLOUR_LOCK") else None)
C.reteam(video, rows, ident)
ident = [I for I in ident if I.get("team") in ("W", "B")]
W = int(1.0 * sfps)


def at(I, f0, f1):
    pts = [C.foot(I["frames"][f]) for f in range(f0, f1 + 1) if f in I["frames"] and not I["frames"][f].get("interp")]
    return np.array(pts) if len(pts) >= 3 else None


ev = []
for i in range(len(ident)):
    for j in range(i + 1, len(ident)):
        if ident[i]["team"] != ident[j]["team"]: continue
        common = sorted(set(ident[i]["frames"]) & set(ident[j]["frames"]))
        near = [f for f in common if np.linalg.norm(C.foot(ident[i]["frames"][f]) - C.foot(ident[j]["frames"][f])) < 0.6 * (C.height(ident[i]["frames"][f]) + C.height(ident[j]["frames"][f])) / 2]
        if not near: continue
        s = p = near[0]
        for f in near[1:] + [None]:
            if f is None or f - p > int(0.8 * sfps):
                ev.append((i, j, s, p))
                if f is not None: s = f
            if f is not None: p = f
suspect = checked = 0
for i, j, s, e in ev:
    bi, bj, ai, aj = at(ident[i], s - W, s - 1), at(ident[j], s - W, s - 1), at(ident[i], e + 1, e + W), at(ident[j], e + 1, e + W)
    if any(x is None for x in (bi, bj, ai, aj)): continue
    gap = (e - s + 2) / sfps
    vi = (bi[-1] - bi[0]) / (len(bi) / sfps); vj = (bj[-1] - bj[0]) / (len(bj) / sfps)
    pi, pj = bi[-1] + vi * gap, bj[-1] + vj * gap
    keep = np.linalg.norm(ai[0] - pi) + np.linalg.norm(aj[0] - pj)
    swap = np.linalg.norm(ai[0] - pj) + np.linalg.norm(aj[0] - pi)
    checked += 1; suspect += swap < 0.8 * keep
jumps = 0
for I in ident:
    ft = np.array([C.foot(I["frames"][f]) for f in sorted(I["frames"])]); h = np.median([C.height(o) for o in I["frames"].values()])
    jumps += int((np.hypot(*np.diff(ft, axis=0).T) > 2.5 * h).sum())
cnt = [sum(1 for I in ident if fi in I["frames"]) for fi in range(len(rows))]
print(f"PRED_W={C.PRED_W:g}: identities {len(ident)} | same-team crossings {len(ev)} (checked {checked}) | SUSPECT swaps {suspect} | position jumps {jumps} | identified {np.mean(cnt):.2f}/{len(ident)} | unassigned frames {sum(len(f) for f,_ in un)}")
