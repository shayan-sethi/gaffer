#!/usr/bin/env python3
"""Lock identities (grass-free colours), label them W1.. / B1.., and write identities.json + an empty passes.json so
player_metrics.py / player_pages.py work on clips without a ball.   python dump_identities.py <video> <out_dir> <start_s> <seconds> <n>"""
import sys, os, json, numpy as np
video, out, st, sec, n = sys.argv[1:6]
sys.argv = ["clip_ids.py", video, out, st, sec, n]
sys.path.insert(0, "scripts")
import clip_ids as C
meta = json.load(open(out + "/meta.json"))
rows = sorted((json.loads(l) for l in open(out + "/dets.jsonl")), key=lambda r: r["frame"]); sfps = meta["fps"] / meta["stride"]
pf, _ = C.track(rows, C.load_mask(meta), [], sfps)
ident, un, *_ = C.lock_identities(pf, sfps, colourer=C.TrackletColour())
C.reteam(video, rows, ident)
TS = int(os.environ.get("TEAM_SIZE", 0))            # players per side, e.g. 6: the TS identities with the most bib colour are the bib team
if TS and len(ident) == 2 * TS:
    share = []
    for I in ident:
        v = [o["lab"][3] for o in I["frames"].values() if o.get("lab") is not None and len(o["lab"]) >= 4 and not o.get("interp")]
        share.append(float(np.median(v)) if v else -1.0)
    order = np.argsort(-np.array(share))
    top = set(int(k) for k in order[:TS])
    print("bib share per identity (sorted):", [round(share[k] / max(float(os.environ.get("SAT_W", 1)), 1), 2) for k in order], flush=True)
    for k, I in enumerate(ident):
        I["team"] = "W" if k in top else "B"; I["cls"] = I["team"]
LBL = dict(kv.split(":") for kv in os.environ.get("TEAM_LABELS", "").split(",") if ":" in kv)      # e.g. TEAM_LABELS=W:P,B:B  (pink team = P)
team = {k: (I["team"] if I["team"] in ("W", "B") else "?") for k, I in enumerate(ident)}
cnt = {"W": 0, "B": 0, "?": 0}; label = {}
for k in sorted(team, key=lambda k: (team[k], k)):
    cnt[team[k]] += 1; label[k] = f"{LBL.get(team[k], team[k])}{cnt[team[k]]}" if team[k] != "?" else f"X{cnt['?']}"
json.dump({str(k): dict(label=label[k], team=team[k], role="P", rows={str(f): [round(v, 1) for v in o["box"]] + [int(bool(o.get("interp")))] for f, o in I["frames"].items()})
           for k, I in enumerate(ident)}, open(out + "/identities.json", "w"))
json.dump({"players": {}}, open(out + "/passes.json", "w"))
print("identities:", {t: sum(1 for v in team.values() if v == t) for t in ("W", "B", "?")}, "| mean identified per frame:",
      round(float(np.mean([sum(1 for I in ident if fi in I["frames"]) for fi in range(len(rows))])), 2))
