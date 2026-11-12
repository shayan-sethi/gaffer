#!/usr/bin/env python3
"""Roster sheet: for every locked identity four crops from different moments + its bib share, to check by eye that each identity is ONE person.
python roster_sheet.py <out_dir>   -> <out_dir>/roster.jpg"""
import sys, json, cv2, numpy as np
D = sys.argv[1]; meta = json.load(open(D + "/meta.json")); ids = json.load(open(D + "/identities.json"))
rows = sorted((json.loads(l) for l in open(D + "/dets.jsonl")), key=lambda r: r["frame"])


def bib(row, box):
    best = max(rows[row]["dets"], key=lambda d: -abs(d[0] - box[0]) - abs(d[3] - box[3])); return best[16] if len(best) > 16 else 0


cap = cv2.VideoCapture(meta["video"]); sheet = []
for I in sorted(ids.values(), key=lambda I: (I["team"], I["label"])):
    rs = sorted(int(k) for k, b in I["rows"].items() if b[3] - b[1] >= 70)
    sh = [bib(r, I["rows"][str(r)]) for r in rs[::5]]
    pick = [rs[int(q * (len(rs) - 1))] for q in (0.05, 0.35, 0.65, 0.95)] if rs else []
    tiles = []
    for r in pick:
        cap.set(cv2.CAP_PROP_POS_FRAMES, meta["start_frame"] + r * meta["stride"]); ok, im = cap.read(); b = I["rows"][str(r)]
        h = b[3] - b[1]; cx = (b[0] + b[2]) / 2; x0 = int(max(0, cx - h * 0.45)); y0 = int(max(0, b[1] - h * 0.05))
        tiles.append(cv2.resize(im[y0:int(b[3] + h * 0.05), x0:int(cx + h * 0.45)], (110, 220)))
    lab = np.zeros((220, 150, 3), np.uint8)
    for k, (txt, sc, y) in enumerate([(I["label"], 1.1, 40), ("bib share", .5, 80), (f"median {np.median(sh):.2f}", .55, 105), (f"frames>0.3: {np.mean(np.array(sh) > 0.3) * 100:.0f}%", .5, 130), (f"{len(rs)} rows", .5, 155)]):
        cv2.putText(lab, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), 2 if k == 0 else 1)
    sheet.append(np.hstack([lab] + tiles))
half = (len(sheet) + 1) // 2
cv2.imwrite(D + "/roster.jpg", np.hstack([np.vstack(sheet[:half]), np.vstack(sheet[half:] + [np.zeros_like(sheet[0])] * (half - len(sheet[half:])))]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print("wrote", D + "/roster.jpg")
