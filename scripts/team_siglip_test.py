#!/usr/bin/env python3
"""
Test the roboflow/sports TeamClassifier recipe (SigLIP embeddings -> UMAP -> KMeans k=2) on torso crops of locked identities,
and compare with the bib-share team labels in identities.json.   python team_siglip_test.py <video> <out_dir> [crops_per_identity=25]
"""
import sys, json
import numpy as np, cv2, torch, umap
from sklearn.cluster import KMeans
from transformers import AutoProcessor, SiglipVisionModel

video, D = sys.argv[1], sys.argv[2]; PER = int(sys.argv[3]) if len(sys.argv) > 3 else 25
ids = json.load(open(D + "/identities.json")); meta = json.load(open(D + "/meta.json"))
rows = sorted((json.loads(l) for l in open(D + "/dets.jsonl")), key=lambda r: r["frame"])
rowframe = {i: r["frame"] for i, r in enumerate(rows)}
sel = {}
for l, I in ids.items():
    fr = sorted(int(f) for f, b in I["rows"].items() if not b[4] and b[3] - b[1] >= 45)
    if fr:
        for f in [fr[i] for i in np.linspace(0, len(fr) - 1, min(PER, len(fr))).astype(int)]:
            sel.setdefault(f, []).append(l)
cap = cv2.VideoCapture(video); crops = []; owner = []
for f in sorted(sel):
    cap.set(cv2.CAP_PROP_POS_FRAMES, rowframe.get(f, f)); ok, im = cap.read()
    if not ok:
        continue
    for l in sel[f]:
        x1, y1, x2, y2 = [int(v) for v in ids[l]["rows"][str(f)][:4]]
        h = y2 - y1; c = im[max(0, y1 + int(.15 * h)):y1 + int(.55 * h), max(0, x1):x2]      # torso only
        if c.size:
            crops.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)); owner.append(l)
print(len(crops), "torso crops from", len(set(owner)), "identities")
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoProcessor.from_pretrained("google/siglip-base-patch16-224", local_files_only=True)
mod = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224", local_files_only=True).to(dev).eval()
E = []
with torch.no_grad():
    for i in range(0, len(crops), 32):
        inp = proc(images=crops[i:i + 32], return_tensors="pt").to(dev)
        E.append(mod(**inp).last_hidden_state.mean(1).cpu().numpy())
E = np.concatenate(E)
P = umap.UMAP(n_components=3, random_state=0).fit_transform(E)
lab = KMeans(2, n_init=10, random_state=0).fit_predict(P)
team = {l: I["team"] for l, I in ids.items()}
print("per identity: share of its crops in cluster 1 | bib-based team")
vote = {}
for l in sorted(team, key=lambda l: (team[l], l)):
    m = [lab[i] for i, o in enumerate(owner) if o == l]
    if m:
        vote[l] = float(np.mean(m)); print(f"  {l:>3s}  {vote[l]:.2f}  {team[l]}")
c1 = {t: np.mean([v for l, v in vote.items() if team[l] == t]) for t in "WB"}
flip = c1["W"] > c1["B"]
bad = [l for l, v in vote.items() if (round(v) == 1) != ((team[l] == "W") == flip)]
print("per-crop agreement with bib labels:", round(float(np.mean([(lab[i] == 1) == ((team[o] == "W") == flip) for i, o in enumerate(owner)])), 3))
print("identities whose majority cluster disagrees with the bib team:", bad)
