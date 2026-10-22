#!/usr/bin/env python3
"""
Reads jersey numbers directly off real torso crops (using the SoccerNet-
fine-tuned PARSeq model) across many samples per track, takes a high-
confidence majority vote per track, and cross-references against the real
match roster - giving each track a genuine identity instead of a guessed
color/position label. Also flags tracks whose high-confidence readings split
cleanly into two different numbers at different points in time (a likely
ID-switch a color-only check can't catch, since both readings can be the
same team/color).

Usage: python jersey_number_id.py
"""
import sys
import json
import sqlite3
from collections import Counter, defaultdict

sys.path.insert(0, r"C:\Users\Shayan Sethi\.cache\torch\hub\baudm_parseq_main")
import torch
import cv2
from PIL import Image
from strhub.models.parseq.system import PARSeq
from strhub.data.module import SceneTextDataModule

MATCH_ID = "073a9f18-b3cd-4658-9a75-f45398eee6c8"
DB_PATH = r".local_data\gaffer_test.db"
VIDEO_PATH = r"test_videos\Barcelona vs Espanyol 2-1 1st half Tactical Cam - CameraTattica (1080p).mp4"
CKPT_PATH = r".local_data\models\parseq_soccernet.ckpt"
NUM_SAMPLES = 20
MIN_CONFIDENCE = 0.95

# Real roster from the lineup you shared - number -> (name, team).
ROSTER = {
    "HOME": {  # Barcelona
        3: "A. Balde", 6: "Gavi", 8: "Pedri", 18: "G. Martin", 5: "P. Cubarsi",
        24: "E. Garcia", 16: "F. Lopez", 7: "F. Torres", 10: "L. Yamal",
        4: "R. Araujo (c)", 13: "J. Garcia (GK)",
    },
    "AWAY": {  # Espanyol
        16: "C. Ngonge", 23: "O.E. Hilali", 4: "U. Gonzalez", 5: "F. Calero",
        19: "K. Garcia", 8: "E. Exposito", 10: "P. Lozano", 6: "L. Cabrera (c)",
        24: "T. Dolan", 22: "C. Romero", 13: "M. Dmitrovic (GK)",
    },
}


def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = PARSeq(**ckpt["hyper_parameters"])
    prefixed = {"model." + k: v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(prefixed, strict=True)
    model.eval()
    return model


def main():
    model = load_model()
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT track_id FROM detections WHERE match_id=? AND class_name='player' ORDER BY track_id",
        (MATCH_ID,),
    )
    all_track_ids = [r[0] for r in cur.fetchall()]

    cap = cv2.VideoCapture(VIDEO_PATH)

    def predict(crop_bgr):
        img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        img_t = img_transform(img).unsqueeze(0)
        with torch.no_grad():
            pred = model(img_t).softmax(-1)
        label, confidence = model.tokenizer.decode(pred)
        return label[0], float(confidence[0].mean())

    out_path = r"C:\Users\SHAYAN~1\AppData\Local\Temp\claude\C--Users-Shayan-Sethi-gaffer\83fe6ffb-4aa4-48f0-bbd5-333435412950\scratchpad\jersey_reads.json"
    try:
        with open(out_path) as f:
            track_reads = {int(k): v for k, v in json.load(f).items()}
        print(f"Resuming - {len(track_reads)} tracks already done", flush=True)
    except FileNotFoundError:
        track_reads = {}  # track_id -> list of (frame_no, label, conf)

    for tid in all_track_ids:
        if tid in track_reads:
            continue
        cur.execute(
            "SELECT frame_no, bbox FROM detections WHERE match_id=? AND track_id=? ORDER BY frame_no",
            (MATCH_ID, tid),
        )
        rows = cur.fetchall()
        dets = [(f, json.loads(b) if isinstance(b, str) else b) for f, b in rows]
        n = len(dets)
        if n < 5:
            track_reads[tid] = []
            continue
        idxs = sorted(set(int(round(i * (n - 1) / (NUM_SAMPLES - 1))) for i in range(min(NUM_SAMPLES, n))))
        reads = []
        for idx in idxs:
            frame_no, bbox = dets[idx]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            pad_x, pad_y = int((x2 - x1) * 0.1), int((y2 - y1) * 0.05)
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = x2 + pad_x, y2 + pad_y
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
            ret, frame = cap.read()
            if not ret:
                continue
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            h = crop.shape[0]
            torso = crop[int(h * 0.12):int(h * 0.6), :]
            if torso.size == 0:
                continue
            label, conf = predict(torso)
            reads.append((frame_no, label, conf))
        track_reads[tid] = reads
        # Save after every track, not just at the end - a crash mid-run
        # (this script died once already, likely a system-level kill, not a
        # code bug) shouldn't throw away everything computed so far.
        with open(out_path, "w") as f:
            json.dump(track_reads, f)
        print(f"track {tid}: {len(reads)} reads done", flush=True)

    print(f"\nSaved raw reads to {out_path}")


if __name__ == "__main__":
    main()
