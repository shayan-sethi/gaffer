#!/usr/bin/env python3
"""
Crossing log and swap-risk check.   python ids_crossings.py <video> <out_dir> <start_s> <seconds> <n>
A crossing = two same-team identities whose feet are within 0.6 body-heights. After each crossing we compare
who-looks-like-whom before and after (SigLIP): if keeping the labels is not clearly better than swapping them,
the crossing is UNCERTAIN (a possible identity swap). Runs the lock without and with the appearance cue and compares.
"""
import sys, json, time
import numpy as np
video, out, st, sec, n = sys.argv[1:6]
sys.argv = ["clip_ids.py", video, out, st, sec, n]
sys.path.insert(0, "scripts")
import clip_ids as C

meta = json.load(open(out + "/meta.json"))
rows = sorted((json.loads(l) for l in open(out + "/dets.jsonl")), key=lambda r: r["frame"])
sfps = meta["fps"] / meta["stride"]
pf, _ = C.track(rows, np.ones((meta["height"], meta["width"]), np.uint8), [], sfps)
emb = C.SigEmbedder(video, rows, per=6)


def crossings(ident):
    ev = []
    for i in range(len(ident)):
        for j in range(i + 1, len(ident)):
            if ident[i]["team"] != ident[j]["team"] or "OFFICIAL" in (ident[i]["role"], ident[j]["role"]):
                continue
            common = sorted(set(ident[i]["frames"]) & set(ident[j]["frames"]))
            near = []
            for f in common:
                a, b = ident[i]["frames"][f], ident[j]["frames"][f]
                h = (C.height(a) + C.height(b)) / 2
                if np.linalg.norm(C.foot(a) - C.foot(b)) < 0.6 * h:
                    near.append(f)
            if not near:
                continue
            s, prev = near[0], near[0]
            for f in near[1:] + [None]:
                if f is None or f - prev > int(1.0 * sfps):
                    if prev - s >= 1:
                        ev.append((i, j, s, prev))
                    if f is not None:
                        s = f
                if f is not None:
                    prev = f
    return ev


def margin(ident, i, j, s, e):
    w = int(2 * sfps)
    def seq(k, a, b):
        return [(f, o) for f, o in sorted(ident[k]["frames"].items()) if a <= f <= b]
    emb.per = 3
    ib, jb = emb(seq(i, s - w, s - 1)), emb(seq(j, s - w, s - 1))
    ia, ja = emb(seq(i, e + 1, e + w)), emb(seq(j, e + 1, e + w))
    emb.per = 6
    if any(v is None for v in (ib, jb, ia, ja)):
        return None
    return float(ib @ ia + jb @ ja - ib @ ja - jb @ ia)


def report(name, embedder):
    t0 = time.time()
    ident, un, *_ = C.lock_identities(pf, sfps, embedder)
    ident = [I for I in ident if I["role"] != "OFFICIAL"]
    ev = crossings(ident)
    ms = [margin(ident, i, j, s, e) for i, j, s, e in ev]
    scored = [m for m in ms if m is not None]
    unc = sum(1 for m in scored if m < 0.04)
    cnt = [sum(1 for I in ident if fi in I["frames"]) for fi in range(len(rows))]
    jumps = 0
    for I in ident:
        ft = np.array([C.foot(I["frames"][f]) for f in sorted(I["frames"])]); h = np.median([C.height(o) for o in I["frames"].values()])
        jumps += int((np.hypot(*np.diff(ft, axis=0).T) > 2.5 * h).sum())
    print(f"{name:34s} crossings {len(ev):3d} | uncertain {unc:3d} of {len(scored):3d} scored | jumps {jumps} | "
          f"identified {np.mean(cnt):.2f}/{len(ident)} | >=N-2 {100*np.mean([c >= len(ident)-2 for c in cnt]):.0f}% | unassigned frames {sum(len(f) for f,_ in un)} ({time.time()-t0:.0f}s)")
    return ident, ev, ms


a = report("without appearance cue", None)
b = report("with appearance cue (SigLIP)", emb)
json.dump({"without": [(int(i), int(j), int(s), int(e), m) for (i, j, s, e), m in zip(a[1], a[2])],
           "with": [(int(i), int(j), int(s), int(e), m) for (i, j, s, e), m in zip(b[1], b[2])]},
          open(out + "/crossings.json", "w"))
