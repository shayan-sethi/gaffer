#!/usr/bin/env python3
"""
The final consolidation step: densely reads jersey numbers across every
included track, finds where WITHIN a track its number actually changes
(a real ID-switch, not just a misread), splits the track's position history
there, then merges any fragments (original tracks or new splits) that agree
on the same (team, number) into one real player identity. Recomputes
PASS/SHOT events on the consolidated identities using the real EventDetector.

Usage: python jersey_split_merge.py <match_id> <pickle_path>
"""
import sys
import json
import pickle
import sqlite3
import bisect
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "backend")
sys.path.insert(0, r"C:\Users\Shayan Sethi\.cache\torch\hub\baudm_parseq_main")

import cv2
import numpy as np
import torch
from PIL import Image
from strhub.models.parseq.system import PARSeq
from strhub.data.module import SceneTextDataModule

from app.cv.tracker import TrackedObject
from app.cv.events import EventDetector
from app.cv.calibration import PitchCalibration

MATCH_ID = sys.argv[1]
PICKLE_PATH = sys.argv[2]
DB_PATH = r".local_data/gaffer_test.db"
VIDEO_PATH = r"test_videos\Barcelona vs Espanyol 2-1 1st half Tactical Cam - CameraTattica (1080p).mp4"
CKPT_PATH = r".local_data/models/parseq_soccernet.ckpt"
BOUNDARY_RESULTS_PATH = r"C:\Users\SHAYAN~1\AppData\Local\Temp\claude\C--Users-Shayan-Sethi-gaffer\83fe6ffb-4aa4-48f0-bbd5-333435412950\scratchpad\pitch_boundary_results.json"
SIGLIP_ASSIGNMENT_PATH = r"C:\Users\SHAYAN~1\AppData\Local\Temp\claude\C--Users-Shayan-Sethi-gaffer\83fe6ffb-4aa4-48f0-bbd5-333435412950\scratchpad\siglip_team_assignment.json"
READS_OUT_PATH = r"C:\Users\SHAYAN~1\AppData\Local\Temp\claude\C--Users-Shayan-Sethi-gaffer\83fe6ffb-4aa4-48f0-bbd5-333435412950\scratchpad\dense_jersey_reads.json"
NUM_SAMPLES = 40
MIN_CONFIDENCE = 0.95
MIN_RUN_LENGTH = 2  # a number needs to repeat at least this many times in a row to count as a real segment, not a misread

ROSTER = {
    "HOME": {3: "A.Balde", 6: "Gavi", 8: "Pedri", 18: "G.Martin", 5: "P.Cubarsi",
              24: "E.Garcia", 16: "F.Lopez", 7: "F.Torres", 10: "L.Yamal",
              4: "R.Araujo", 13: "J.Garcia(GK)"},
    "AWAY": {16: "C.Ngonge", 23: "O.Hilali", 4: "U.Gonzalez", 5: "F.Calero",
              19: "K.Garcia", 8: "E.Exposito", 10: "P.Lozano", 6: "L.Cabrera",
              24: "T.Dolan", 22: "C.Romero", 13: "M.Dmitrovic(GK)"},
}


def load_parseq():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = PARSeq(**ckpt["hyper_parameters"])
    prefixed = {"model." + k: v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(prefixed, strict=True)
    model.eval()
    return model


def main():
    with open(BOUNDARY_RESULTS_PATH) as f:
        boundary = json.load(f)
    included = sorted([int(tid) for tid, r in boundary.items() if r["is_player"]])
    print(f"included tracks: {len(included)}: {included}")

    with open(SIGLIP_ASSIGNMENT_PATH) as f:
        siglip = json.load(f)
    track_team: Dict[int, str] = {}
    for tid in siglip["included_home"]:
        track_team[tid] = "HOME"
    for tid in siglip["included_away"]:
        track_team[tid] = "AWAY"

    model = load_parseq()
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    def predict(crop_bgr):
        img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        img_t = img_transform(img).unsqueeze(0)
        with torch.no_grad():
            pred = model(img_t).softmax(-1)
        label, confidence = model.tokenizer.decode(pred)
        return label[0], float(confidence[0].mean())

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cap = cv2.VideoCapture(VIDEO_PATH)

    try:
        with open(READS_OUT_PATH) as f:
            track_reads = {int(k): v for k, v in json.load(f).items()}
    except FileNotFoundError:
        track_reads = {}

    for tid in included:
        if tid in track_reads:
            continue
        cur.execute(
            "SELECT frame_no, bbox FROM detections WHERE match_id=? AND track_id=? ORDER BY frame_no",
            (MATCH_ID, tid),
        )
        rows = cur.fetchall()
        dets = [(f, json.loads(b) if isinstance(b, str) else b) for f, b in rows]
        n = len(dets)
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
        with open(READS_OUT_PATH, "w") as f:
            json.dump(track_reads, f)
        print(f"track {tid}: {len(reads)} reads done", flush=True)

    # --- Segment each track: find real, sustained number changes over time ---
    def valid_number(label, conf, team):
        if conf < MIN_CONFIDENCE or not label.isdigit():
            return None
        num = int(label)
        return num if num in ROSTER.get(team, {}) else None

    track_segments: Dict[int, List[Tuple[int, int, Optional[int]]]] = {}  # tid -> [(start_frame, end_frame, number)]
    for tid in included:
        team = track_team.get(tid, "HOME")
        reads = sorted(track_reads[tid], key=lambda r: r[0])
        valid = [(f, valid_number(l, c, team)) for f, l, c in reads]
        valid = [(f, n) for f, n in valid if n is not None]
        if not valid:
            track_segments[tid] = [(0, 10**9, None)]
            continue

        # run-length group consecutive (in read-order) same numbers, dropping
        # isolated single misreads that don't repeat.
        runs = []
        cur_num, cur_start = valid[0][1], valid[0][0]
        cur_count = 1
        last_frame = valid[0][0]
        for f, n in valid[1:]:
            if n == cur_num:
                cur_count += 1
                last_frame = f
            else:
                runs.append((cur_start, last_frame, cur_num, cur_count))
                cur_num, cur_start, cur_count = n, f, 1
                last_frame = f
        runs.append((cur_start, last_frame, cur_num, cur_count))

        # keep only runs with enough repeats to trust, merge adjacent same-number runs
        strong_runs = [r for r in runs if r[3] >= MIN_RUN_LENGTH]
        if not strong_runs:
            # nothing repeated enough - fall back to simple majority vote, no split
            majority = Counter(n for _, n in valid).most_common(1)[0][0]
            track_segments[tid] = [(0, 10**9, majority)]
            continue

        merged = [list(strong_runs[0])]
        for r in strong_runs[1:]:
            if r[2] == merged[-1][2]:
                merged[-1][1] = r[1]
                merged[-1][3] += r[3]
            else:
                merged.append(list(r))

        if len(merged) == 1:
            track_segments[tid] = [(0, 10**9, merged[0][2])]
        else:
            # real split(s): boundary is the midpoint between consecutive runs' frames
            segs = []
            prev_end = 0
            for i, r in enumerate(merged):
                if i + 1 < len(merged):
                    boundary = (r[1] + merged[i + 1][0]) // 2
                else:
                    boundary = 10**9
                segs.append((prev_end, boundary, r[2]))
                prev_end = boundary
            track_segments[tid] = segs
            print(f"track {tid}: SPLIT into {len(segs)} segments: {[(s[2]) for s in segs]}")

    with open(r"C:\Users\SHAYAN~1\AppData\Local\Temp\claude\C--Users-Shayan-Sethi-gaffer\83fe6ffb-4aa4-48f0-bbd5-333435412950\scratchpad\track_segments.json", "w") as f:
        json.dump({str(k): v for k, v in track_segments.items()}, f, indent=2)

    # --- Merge segments sharing the same (team, number) into one identity ---
    with open(PICKLE_PATH, "rb") as f:
        payload = pickle.load(f)
    raw_tracks = payload["tracks"]

    def get_history(tid):
        # A track that a live pipeline run's OWN (buggy) team-assignment
        # excluded as "REFEREE" never makes it into the pickle's tracks dict
        # at all (see track.py's serializable_tracks) - even though our
        # independent boundary/SigLIP checks confirmed it's a real player.
        # Reconstruct its position history directly from the raw detections
        # instead of silently losing it.
        if str(tid) in raw_tracks:
            return [(f, tuple(pt)) for f, pt in raw_tracks[str(tid)]["history"]]
        cur.execute(
            "SELECT frame_no, bbox FROM detections WHERE match_id=? AND track_id=? ORDER BY frame_no",
            (MATCH_ID, tid),
        )
        rows = cur.fetchall()
        hist = []
        for f, bbox_json in rows:
            bbox = json.loads(bbox_json) if isinstance(bbox_json, str) else bbox_json
            hist.append((f, ((bbox[0] + bbox[2]) / 2.0, bbox[3])))
        return hist

    # Each candidate piece: (source_track_id, start_frame, end_frame, seg_hist, n_supporting_reads)
    Piece = Tuple[int, int, int, List[Tuple[int, Tuple[float, float]]], int]
    identity_segments: Dict[Tuple[str, Optional[int]], List[Piece]] = defaultdict(list)
    for tid in included:
        team = track_team.get(tid, "HOME")
        history = get_history(tid)
        n_reads_for_tid = len(track_reads.get(tid, []))
        for start_f, end_f, num in track_segments[tid]:
            seg_hist = [(f, tuple(pt)) for f, pt in history if start_f <= f < end_f]
            if not seg_hist:
                continue
            key = (team, num)
            identity_segments[key].append((tid, start_f, end_f, seg_hist, n_reads_for_tid))

    def overlaps(a: Piece, b: Piece) -> bool:
        return a[1] < b[2] and b[1] < a[2]

    print(f"\nresolving {len(identity_segments)} candidate (team, number) groups from {len(included)} tracks:")
    consolidated: Dict[str, TrackedObject] = {}
    next_id = 9000
    unconfirmed_count = 0
    for (team, num), pieces in identity_segments.items():
        name = ROSTER.get(team, {}).get(num, "?") if num else "(number unconfirmed)"
        if len(pieces) == 1 or num is None:
            groups = [[p] for p in pieces]
        else:
            # Two segments claiming the same number can only be the same real
            # person if they were never active at the same time - a real
            # player can't be in two places at once. Where pieces conflict
            # (overlap), that's decisive proof the number reading is wrong
            # for at least one of them (most likely model hallucination on
            # ambiguous frames - see MIN_RUN_LENGTH's limits), so they must
            # NOT be merged: keep the piece with the most supporting reads
            # under this identity, and let every piece it conflicts with fall
            # back to its own separate, number-unconfirmed identity instead
            # of being silently (and wrongly) fused together.
            pieces_sorted = sorted(pieces, key=lambda p: -p[4])
            accepted: List[Piece] = []
            rejected: List[Piece] = []
            for p in pieces_sorted:
                if any(overlaps(p, a) for a in accepted):
                    rejected.append(p)
                else:
                    accepted.append(p)
            groups = [accepted] + [[p] for p in rejected]
            if rejected:
                print(f"  CONFLICT on {team} #{num}: kept track {accepted[0][0]} (+{len(accepted)-1} non-overlapping), "
                      f"rejected {sorted(set(p[0] for p in rejected))} as separate unconfirmed identities")

        # groups[0] is the accepted (non-conflicting) merge and keeps the
        # number; every other group lost a conflict and becomes its own
        # separate, number-unconfirmed identity instead of being merged.
        for i, group in enumerate(groups):
            keeps_number = num is not None and i == 0
            source_tids = sorted(set(p[0] for p in group))
            all_hist = []
            for _, _, _, seg_hist, _ in group:
                all_hist.extend(seg_hist)
            if not all_hist:
                continue
            all_hist.sort(key=lambda x: x[0])
            obj_id = next_id
            next_id += 1
            obj = TrackedObject(track_id=obj_id, class_name="player", bbox=[0, 0, 0, 0], centroid=all_hist[-1][1], frame_no=0)
            obj.team = team
            obj.history = all_hist
            final_num = num if keeps_number else None
            tag = f"{team}_{final_num if final_num is not None else 'unconf' + str(obj_id)}"
            if not keeps_number and num is not None:
                unconfirmed_count += 1
            display_name = ROSTER.get(team, {}).get(final_num, "?") if final_num else (
                "(unconfirmed - lost number conflict)" if num is not None else "(number unconfirmed)"
            )
            print(f"  {team} #{final_num if final_num is not None else '?'} {display_name}: from tracks {source_tids}")
            consolidated[tag] = obj

    print(f"\nFINAL PLAYER COUNT: {len(consolidated)} (target: 22)")

    # --- Run the real EventDetector on the consolidated identities + ball ---
    total_frames = payload.get("total_frames", 150)
    fps = payload.get("fps", 30.0)
    width = payload.get("frame_width", 1280)
    height = payload.get("frame_height", 720)
    pitch_bounds = payload.get("pitch_bounds")
    pitch_bounds_history = payload.get("pitch_bounds_history") or []
    pbh_frames = [f for f, _ in pitch_bounds_history]
    home_attacks_right = payload.get("home_attacks_right", True)
    goal_axis = payload.get("goal_axis", "x")
    pitch_length_m = payload.get("pitch_length_m", 40.0)
    pitch_width_m = payload.get("pitch_width_m", 20.0)
    calibration = PitchCalibration.from_points_json(
        payload.get("calibration_points"), pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m
    )

    ball_data = next(t for t in raw_tracks.values() if t["class_name"] == "ball")
    ball_obj = TrackedObject(track_id=26, class_name="ball", bbox=[0, 0, 0, 0], centroid=(0, 0), frame_no=0)
    ball_obj.history = [(f, tuple(pt)) for f, pt in ball_data["history"]]
    consolidated["ball"] = ball_obj

    track_positions = {}
    track_frames_sorted = {}
    id_to_label = {}
    for label, obj in consolidated.items():
        positions_by_frame = {f: pt for f, pt in obj.history}
        track_positions[obj.track_id] = positions_by_frame
        track_frames_sorted[obj.track_id] = sorted(positions_by_frame.keys())
        id_to_label[obj.track_id] = label

    max_gap_ball = int(fps * 3)
    max_gap_player = int(fps * 1)

    def position_at_frame(frames_sorted, positions, frame_no, max_gap):
        if frame_no in positions:
            return positions[frame_no]
        i = bisect.bisect_left(frames_sorted, frame_no)
        candidates = []
        if i > 0:
            candidates.append(frames_sorted[i - 1])
        if i < len(frames_sorted):
            candidates.append(frames_sorted[i])
        if not candidates:
            return None
        nearest = min(candidates, key=lambda f: abs(f - frame_no))
        if abs(nearest - frame_no) > max_gap:
            return None
        return positions[nearest]

    event_detector = EventDetector(
        frame_width=width, frame_height=height, fps=fps,
        pitch_bounds=tuple(pitch_bounds) if pitch_bounds else None,
        pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m,
        calibration=calibration, home_attacks_right=home_attacks_right, goal_axis=goal_axis,
    )

    all_events = []
    for frame_no in range(1, total_frames + 1):
        frame_tracks = []
        for label, obj in consolidated.items():
            max_gap = max_gap_ball if obj.class_name == "ball" else max_gap_player
            pt = position_at_frame(track_frames_sorted[obj.track_id], track_positions[obj.track_id], frame_no, max_gap)
            if pt is None:
                continue
            current_t = TrackedObject(
                track_id=obj.track_id, class_name=obj.class_name,
                bbox=[pt[0] - 20, pt[1] - 40, pt[0] + 20, pt[1]], centroid=pt, frame_no=frame_no,
            )
            current_t.team = obj.team
            frame_tracks.append(current_t)

        if pbh_frames:
            idx2 = min(bisect.bisect_right(pbh_frames, frame_no), len(pbh_frames)) - 1
            if idx2 >= 0:
                event_detector.update_pitch_bounds(tuple(pitch_bounds_history[idx2][1]))

        frame_evts = event_detector.process_frame(frame_tracks, frame_no)
        all_events.extend(frame_evts)

    print(f"\n{len(all_events)} total events")
    by_type_team = Counter((e.event_type, e.team) for e in all_events)
    for (etype, team), count in sorted(by_type_team.items()):
        print(f"  {etype:20} {team:8} {count}")

    print("\nPer-player pass counts:")
    pass_counts = Counter()
    for e in all_events:
        if e.event_type == "PASS":
            pass_counts[id_to_label.get(e.player_track_id, e.player_track_id)] += 1
    for label, count in pass_counts.most_common():
        print(f"  {label}: {count}")

    print("\nSHOT events:")
    for e in all_events:
        if e.event_type == "SHOT":
            print(f"  [{e.frame_no/fps:.1f}s] {e.team} {e.data_json}")


if __name__ == "__main__":
    main()
