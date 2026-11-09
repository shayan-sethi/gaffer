#!/usr/bin/env python3
"""
Reproduces the Roboflow "Football AI Tutorial" pipeline on one clip, to
confirm the setup works before building on it:
  4-class detector (ball/goalkeeper/player/referee) -> ByteTrack ->
  SigLIP+UMAP+KMeans team split -> goalkeepers go to the nearest team.

Usage: python roboflow_tutorial_repro.py <video> <out_dir>
"""
import sys
import os
import json
from collections import defaultdict

import numpy as np
import supervision as sv
from ultralytics import YOLO
from sports.common.team import TeamClassifier

VIDEO = sys.argv[1]
OUT_DIR = sys.argv[2]
os.makedirs(OUT_DIR, exist_ok=True)

PLAYER_MODEL = r".local_data\models\football-player-detection.pt"
BALL, GOALKEEPER, PLAYER, REFEREE = 0, 1, 2, 3
CONF = 0.3
STRIDE = 30  # frames sampled to fit the team classifier
TEAM_REFRESH = 25  # re-predict a track's team this often (frames)


def main():
    model = YOLO(PLAYER_MODEL)
    info = sv.VideoInfo.from_video_path(VIDEO)
    print(f"video: {info.width}x{info.height} {info.fps}fps {info.total_frames} frames", flush=True)

    # --- 1. fit team classifier on player crops from every STRIDE-th frame ---
    crops = []
    for frame in sv.get_video_frames_generator(VIDEO, stride=STRIDE):
        det = sv.Detections.from_ultralytics(model(frame, imgsz=1280, verbose=False)[0])
        det = det[det.confidence > CONF]
        players = det[det.class_id == PLAYER]
        crops += [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
    print(f"fitting TeamClassifier on {len(crops)} player crops", flush=True)
    team_clf = TeamClassifier(device="cpu")
    team_clf.fit(crops)

    # --- 2. full pass: detect, track, assign teams ---
    tracker = sv.ByteTrack(frame_rate=int(info.fps))
    ellipse = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]), thickness=2)
    label = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
                              text_color=sv.Color.from_hex("#000000"), text_position=sv.Position.BOTTOM_CENTER)
    tri = sv.TriangleAnnotator(color=sv.Color.from_hex("#FFD700"), base=25, height=21, outline_thickness=1)

    team_cache = {}  # tracker_id -> (team, frame_last_predicted)
    ids_seen = set()
    per_frame_counts = []
    track_team_votes = defaultdict(list)
    out_video = os.path.join(OUT_DIR, "annotated.mp4")
    still_frames = {0, info.total_frames // 2, info.total_frames - 2}

    frame_log = open(os.path.join(OUT_DIR, "frames.jsonl"), "w")
    with sv.VideoSink(out_video, info) as sink:
        for i, frame in enumerate(sv.get_video_frames_generator(VIDEO)):
            det = sv.Detections.from_ultralytics(model(frame, imgsz=1280, verbose=False)[0])
            ball = det[det.class_id == BALL]
            ball.xyxy = sv.pad_boxes(xyxy=ball.xyxy, px=10)
            rest = det[det.class_id != BALL].with_nms(threshold=0.5, class_agnostic=True)
            rest = tracker.update_with_detections(rest)

            gk = rest[rest.class_id == GOALKEEPER]
            pl = rest[rest.class_id == PLAYER]
            ref = rest[rest.class_id == REFEREE]

            # CPU shortcut: the tutorial predicts every crop every frame on a
            # GPU. Here each track's team is predicted when first seen and
            # refreshed every TEAM_REFRESH frames - a player doesn't change
            # kit, so this gives the same labels for a fraction of the compute.
            team_ids = np.array([], dtype=int)
            if len(pl):
                stale = [k for k, tid in enumerate(pl.tracker_id)
                         if int(tid) not in team_cache or i - team_cache[int(tid)][1] >= TEAM_REFRESH]
                if stale:
                    preds = team_clf.predict([sv.crop_image(frame, pl.xyxy[k]) for k in stale])
                    for k, p in zip(stale, preds):
                        team_cache[int(pl.tracker_id[k])] = (int(p), i)
                team_ids = np.array([team_cache[int(tid)][0] for tid in pl.tracker_id], dtype=int)
            gk_team = np.array([], dtype=int)
            if len(gk) and len(pl):
                gk_xy = gk.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                pl_xy = pl.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                c0 = pl_xy[team_ids == 0].mean(axis=0) if (team_ids == 0).any() else None
                c1 = pl_xy[team_ids == 1].mean(axis=0) if (team_ids == 1).any() else None
                for g in gk_xy:
                    if c0 is None or c1 is None:
                        gk_team = np.append(gk_team, 0)
                    else:
                        gk_team = np.append(gk_team, 0 if np.linalg.norm(g - c0) < np.linalg.norm(g - c1) else 1)

            for tid, t in zip(pl.tracker_id, team_ids):
                track_team_votes[int(tid)].append(int(t))
            ids_seen.update(int(t) for t in rest.tracker_id)
            per_frame_counts.append((len(pl), len(gk), len(ref)))
            # flushed every frame so a crash mid-run never loses results
            frame_log.write(json.dumps({
                "frame": i,
                "players": [[int(t), int(c)] for t, c in zip(pl.tracker_id, team_ids)],
                "goalkeepers": [[int(t), int(c)] for t, c in zip(gk.tracker_id, gk_team)],
                "referees": [int(t) for t in ref.tracker_id],
            }) + "\n")
            frame_log.flush()

            # annotate
            all_det = sv.Detections.merge([pl, gk, ref])
            colors = np.concatenate([team_ids, gk_team, np.full(len(ref), 2)]).astype(int) if len(all_det) else np.array([], dtype=int)
            all_det.class_id = colors
            labels = [f"#{t}" for t in all_det.tracker_id]
            out = ellipse.annotate(frame.copy(), all_det)
            out = label.annotate(out, all_det, labels=labels)
            out = tri.annotate(out, ball)
            sink.write_frame(out)
            if i in still_frames:
                import cv2
                cv2.imwrite(os.path.join(OUT_DIR, f"still_{i}.jpg"), out)
            if i % 100 == 0:
                print(f"  frame {i}/{info.total_frames}", flush=True)

    counts = np.array(per_frame_counts)
    flips = sum(1 for v in track_team_votes.values() if len(set(v)) > 1)
    summary = {
        "frames": len(per_frame_counts),
        "distinct_tracker_ids_total": len(ids_seen),
        "mean_players_per_frame": float(counts[:, 0].mean()),
        "mean_goalkeepers_per_frame": float(counts[:, 1].mean()),
        "mean_referees_per_frame": float(counts[:, 2].mean()),
        "max_players_in_a_frame": int(counts[:, 0].max()),
        "player_tracks": len(track_team_votes),
        "player_tracks_with_team_flips": flips,
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
