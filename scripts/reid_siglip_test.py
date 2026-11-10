#!/usr/bin/env python3
"""
Can a general image model (SigLIP, already cached locally) tell same-kit teammates apart?
For each locked identity, take crops from the first half and the last half of the clip. Then:
  * retrieval accuracy: does a last-half crop's nearest first-half identity (same team only) match its own? (chance = 1/team size)
  * mean cosine similarity: same player over time vs different teammates.
python reid_siglip_test.py <video> <out_dir> <start_s> <seconds> <n> [crops_per_half=20]
"""
import sys, json
import numpy as np
import cv2
import torch

video, out, st, sec, n = sys.argv[1:6]
PER = int(sys.argv[6]) if len(sys.argv) > 6 else 20
sys.argv = ["clip_ids.py", video, out, st, sec, n]
sys.path.insert(0, "scripts")
import clip_ids as C
from transformers import AutoModel, AutoProcessor

meta = json.load(open(out + "/meta.json"))
rows = sorted((json.loads(l) for l in open(out + "/dets.jsonl")), key=lambda r: r["frame"])
sfps = meta["fps"] / meta["stride"]
pf, _ = C.track(rows, np.ones((meta["height"], meta["width"]), np.uint8), [], sfps)
ident, *_ = C.lock_identities(pf, sfps)
ident = [I for I in ident if I["role"] != "OFFICIAL"]
print(len(ident), "identities")

def pick(I, half):
    fr = sorted(f for f, o in I["frames"].items() if not o.get("interp") and C.height(o) >= 45)
    mid = len(rows) // 2
    fr = [f for f in fr if (f < mid) == (half == 0)]
    if not fr:
        return []
    return [fr[i] for i in np.linspace(0, len(fr) - 1, min(PER, len(fr))).astype(int)]

sel = {(k, h): pick(I, h) for k, I in enumerate(ident) for h in (0, 1)}
need = sorted({f for v in sel.values() for f in v})
cap = cv2.VideoCapture(video)
crops = {}
for f in need:
    cap.set(cv2.CAP_PROP_POS_FRAMES, rows[f]["frame"]); ok, im = cap.read()
    for k, I in enumerate(ident):
        o = I["frames"].get(f)
        if o is not None and not o.get("interp") and any(f in sel[(k, h)] for h in (0, 1)):
            x1, y1, x2, y2 = [int(v) for v in o["box"]]
            c = im[max(0, y1):y2, max(0, x1):x2]
            if c.size:
                crops[(k, f)] = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
print(len(crops), "crops")

proc = AutoProcessor.from_pretrained("google/siglip-base-patch16-224", local_files_only=True)
model = AutoModel.from_pretrained("google/siglip-base-patch16-224", local_files_only=True).eval()
keys = list(crops)
embs = []
with torch.no_grad():
    for i in range(0, len(keys), 32):
        batch = [crops[k] for k in keys[i:i + 32]]
        inp = proc(images=batch, return_tensors="pt")
        e = model.get_image_features(**inp)
        embs.append(torch.nn.functional.normalize(e, dim=-1).numpy())
E = dict(zip(keys, np.concatenate(embs)))

def mean_emb(k, h):
    v = [E[(k, f)] for f in sel[(k, h)] if (k, f) in E]
    if not v:
        return None
    m = np.mean(v, axis=0)
    return m / np.linalg.norm(m)

M = {(k, h): mean_emb(k, h) for k in range(len(ident)) for h in (0, 1)}
teams = [I["team"] for I in ident]
same, diff_team, diff_cross = [], [], []
correct = tot = 0
for k in range(len(ident)):
    if M[(k, 0)] is None or M[(k, 1)] is None:
        continue
    same.append(float(M[(k, 0)] @ M[(k, 1)]))
    mates = [j for j in range(len(ident)) if j != k and teams[j] == teams[k] and M[(j, 0)] is not None]
    for j in mates:
        diff_team.append(float(M[(k, 1)] @ M[(j, 0)]))
    for j in range(len(ident)):
        if teams[j] != teams[k] and M[(j, 0)] is not None:
            diff_cross.append(float(M[(k, 1)] @ M[(j, 0)]))
    # per-crop retrieval among same-team candidates
    cands = [k] + mates
    for f in sel[(k, 1)]:
        if (k, f) not in E:
            continue
        sims = [float(E[(k, f)] @ M[(c, 0)]) for c in cands]
        correct += int(cands[int(np.argmax(sims))] == k); tot += 1
# how much averaging helps: groups of m consecutive late-half crops of one player vs the teammates' early-half mean
for m in (3, 6, 10):
    ok_ = n_ = 0
    for k in range(len(ident)):
        if M[(k, 0)] is None:
            continue
        mates = [j for j in range(len(ident)) if j != k and teams[j] == teams[k] and M[(j, 0)] is not None]
        cands = [k] + mates
        fs = [f for f in sel[(k, 1)] if (k, f) in E]
        for i in range(0, len(fs) - m + 1, m):
            g = np.mean([E[(k, f)] for f in fs[i:i + m]], axis=0); g /= np.linalg.norm(g)
            ok_ += int(cands[int(np.argmax([float(g @ M[(c, 0)]) for c in cands]))] == k); n_ += 1
    print(f"retrieval using the average of {m} crops: {100 * ok_ / max(n_, 1):.0f}% correct ({n_} groups)")
chance = np.mean([1 / sum(1 for t in teams if t == tm) for tm in teams])
print(f"cosine similarity  same player (early vs late): {np.mean(same):.3f}")
print(f"                   different teammate         : {np.mean(diff_team):.3f}")
print(f"                   other team                 : {np.mean(diff_cross):.3f}")
print(f"retrieval among teammates: {100*correct/tot:.0f}% correct over {tot} crops (chance about {100*chance:.0f}%)")
