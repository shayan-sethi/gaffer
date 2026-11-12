#!/usr/bin/env python3
"""
Follows the Roboflow "Football AI Tutorial" (youtube aBVGKoNZQUw) step by step:

  1. 4-class detector (ball/goalkeeper/player/referee), conf 0.3. Ball boxes
     padded 10px; everything else class-agnostic NMS (iou 0.5).
  2. ByteTrack on everything except the ball.
  3. Team split: SigLIP + UMAP + KMeans, fit on player crops from every 30th
     frame, then PREDICT EVERY FRAME (no caching). Goalkeepers -> nearest team
     centroid (mean bottom-center position of each team's players).
  4. 32-keypoint pitch model, keypoints with conf > 0.5, homography via
     cv2.findHomography against SoccerPitchConfiguration vertices; homography
     matrix averaged over a window of 5.
  5. Project players/goalkeepers/referees/ball to the pitch -> radar and
     Voronoi videos.
  6. Ball path: drop frames with more than one ball, then drop jumps over 5 m.

Writes per-frame results to frames.jsonl as it goes (crash-safe).

Usage: python roboflow_full_pipeline.py <video> <out_dir> [max_frames]
"""
import sys
import os
import json
from collections import deque

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from sports.annotators.soccer import (
    draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram, draw_paths_on_pitch,
)
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

VIDEO = sys.argv[1]
OUT_DIR = sys.argv[2]
MAX_FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else None
os.makedirs(OUT_DIR, exist_ok=True)

PLAYER_MODEL = r".local_data/models/football-player-detection.pt"
PITCH_MODEL = r".local_data/models/football-pitch-detection.pt"
BALL, GOALKEEPER, PLAYER, REFEREE = 0, 1, 2, 3
CONF = 0.3
KEYPOINT_CONF = 0.5
STRIDE = 30
HOMOGRAPHY_WINDOW = 5
BALL_MAX_JUMP_CM = 500  # 5 m, as in the tutorial

CONFIG = SoccerPitchConfiguration()
TEAM_COLORS = ["#00BFFF", "#FF1493", "#FFD700"]


def resolve_goalkeepers_team_id(players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
    gk_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pl_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    t0 = pl_xy[players.class_id == 0].mean(axis=0)
    t1 = pl_xy[players.class_id == 1].mean(axis=0)
    out = []
    for g in gk_xy:
        out.append(0 if np.linalg.norm(g - t0) < np.linalg.norm(g - t1) else 1)
    return np.array(out)


def replace_outliers_based_on_distance(positions, distance_threshold):
    """Drop a detection if it jumped more than the threshold from the last kept one."""
    last_valid = None
    cleaned = []
    for p in positions:
        if len(p) == 0:
            cleaned.append(p)
            continue
        if last_valid is None:
            cleaned.append(p)
            last_valid = p
        elif np.linalg.norm(p - last_valid) > distance_threshold:
            cleaned.append(np.array([], dtype=np.float64))
        else:
            cleaned.append(p)
            last_valid = p
    return cleaned


def main():
    det_model = YOLO(PLAYER_MODEL)
    pitch_model = YOLO(PITCH_MODEL)
    info = sv.VideoInfo.from_video_path(VIDEO)
    total = min(info.total_frames, MAX_FRAMES) if MAX_FRAMES else info.total_frames
    print(f"video {info.width}x{info.height} {info.fps}fps, processing {total} frames", flush=True)

    # --- team classifier: crops from every STRIDE-th frame ---
    crops = []
    for i, frame in enumerate(sv.get_video_frames_generator(VIDEO, stride=STRIDE)):
        if MAX_FRAMES and i * STRIDE >= total:
            break
        det = sv.Detections.from_ultralytics(det_model(frame, imgsz=1280, verbose=False)[0])
        det = det[det.confidence > CONF].with_nms(threshold=0.5, class_agnostic=True)
        crops += [sv.crop_image(frame, xyxy) for xyxy in det[det.class_id == PLAYER].xyxy]
    print(f"fitting TeamClassifier on {len(crops)} crops", flush=True)
    team_clf = TeamClassifier(device="cpu")
    team_clf.fit(crops)

    tracker = sv.ByteTrack(frame_rate=int(info.fps))
    palette = sv.ColorPalette.from_hex(TEAM_COLORS)
    ellipse = sv.EllipseAnnotator(color=palette, thickness=2)
    label = sv.LabelAnnotator(color=palette, text_color=sv.Color.from_hex("#000000"),
                              text_position=sv.Position.BOTTOM_CENTER)
    tri = sv.TriangleAnnotator(color=sv.Color.from_hex("#FFD700"), base=25, height=21, outline_thickness=1)

    matrices = deque(maxlen=HOMOGRAPHY_WINDOW)
    last_transformer = None
    ball_path = []  # per frame: array of ball positions in pitch cm (0, 1 or more)
    pitch_vertices = np.array(CONFIG.vertices, dtype=np.float32)

    probe = draw_pitch(CONFIG)
    radar_info = sv.VideoInfo(width=probe.shape[1], height=probe.shape[0], fps=info.fps, total_frames=total)

    frame_log = open(os.path.join(OUT_DIR, "frames.jsonl"), "w")
    sink_ann = sv.VideoSink(os.path.join(OUT_DIR, "annotated.mp4"), info)
    sink_radar = sv.VideoSink(os.path.join(OUT_DIR, "radar.mp4"), radar_info)
    sink_vor = sv.VideoSink(os.path.join(OUT_DIR, "voronoi.mp4"), radar_info)
    stills = {0, total // 2, total - 2}
    n_kp_hist = []
    ids_seen = set()

    with sink_ann, sink_radar, sink_vor:
        for i, frame in enumerate(sv.get_video_frames_generator(VIDEO)):
            if i >= total:
                break
            # --- detection ---
            det = sv.Detections.from_ultralytics(det_model(frame, imgsz=1280, verbose=False)[0])
            det = det[det.confidence > CONF]
            ball = det[det.class_id == BALL]
            ball.xyxy = sv.pad_boxes(xyxy=ball.xyxy, px=10)
            rest = det[det.class_id != BALL].with_nms(threshold=0.5, class_agnostic=True)
            rest = tracker.update_with_detections(rest)
            gk = rest[rest.class_id == GOALKEEPER]
            pl = rest[rest.class_id == PLAYER]
            ref = rest[rest.class_id == REFEREE]

            # --- teams: predict every frame, exactly as in the video ---
            if len(pl):
                pl.class_id = team_clf.predict([sv.crop_image(frame, x) for x in pl.xyxy])
            if len(gk) and len(pl) and (pl.class_id == 0).any() and (pl.class_id == 1).any():
                gk.class_id = resolve_goalkeepers_team_id(pl, gk)
            elif len(gk):
                gk.class_id = np.zeros(len(gk), dtype=int)
            ref.class_id = np.full(len(ref), 2, dtype=int)
            ids_seen.update(int(t) for t in rest.tracker_id)

            # --- keypoints + homography (matrix averaged over a window) ---
            kp_res = pitch_model(frame, verbose=False)[0]
            kp = sv.KeyPoints.from_ultralytics(kp_res)
            transformer = last_transformer
            n_kp = 0
            if len(kp.xy):
                mask = kp.keypoint_confidence[0] > KEYPOINT_CONF
                n_kp = int(mask.sum())
                if n_kp >= 4:
                    t = ViewTransformer(source=kp.xy[0][mask].astype(np.float32), target=pitch_vertices[mask])
                    matrices.append(t.m)
                    t.m = np.mean(np.array(matrices), axis=0)
                    transformer = t
                    last_transformer = t
            n_kp_hist.append(n_kp)

            # --- project to pitch ---
            def project(d: sv.Detections):
                xy = d.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                return transformer.transform_points(xy) if (transformer is not None and len(xy)) else np.zeros((0, 2))

            pitch_ball = project(ball)
            pitch_pl = project(pl)
            pitch_gk = project(gk)
            pitch_ref = project(ref)
            ball_path.append(pitch_ball)

            allp = np.concatenate([pitch_pl, pitch_gk]) if len(pitch_gk) or len(pitch_pl) else np.zeros((0, 2))
            all_cls = np.concatenate([pl.class_id, gk.class_id]) if len(pl) or len(gk) else np.array([], dtype=int)

            frame_log.write(json.dumps({
                "frame": i, "n_keypoints": n_kp,
                "players": [[int(t), int(c), [round(float(x), 1), round(float(y), 1)]] for t, c, (x, y) in zip(pl.tracker_id, pl.class_id, pitch_pl)],
                "goalkeepers": [[int(t), int(c), [round(float(x), 1), round(float(y), 1)]] for t, c, (x, y) in zip(gk.tracker_id, gk.class_id, pitch_gk)],
                "referees": [[int(t), [round(float(x), 1), round(float(y), 1)]] for t, (x, y) in zip(ref.tracker_id, pitch_ref)],
                "ball": [[round(float(x), 1), round(float(y), 1)] for x, y in pitch_ball],
            }) + "\n")
            frame_log.flush()

            # --- annotated frame ---
            merged = sv.Detections.merge([pl, gk, ref])
            out = ellipse.annotate(frame.copy(), merged)
            out = label.annotate(out, merged, labels=[f"#{t}" for t in merged.tracker_id])
            out = tri.annotate(out, ball)
            sink_ann.write_frame(out)

            # --- radar + voronoi ---
            radar = draw_pitch(CONFIG)
            vor = draw_pitch(CONFIG)
            if len(allp):
                for team_id, col in ((0, "#00BFFF"), (1, "#FF1493")):
                    sel = allp[all_cls == team_id]
                    if len(sel):
                        radar = draw_points_on_pitch(CONFIG, sel, face_color=sv.Color.from_hex(col),
                                                     edge_color=sv.Color.BLACK, radius=16, pitch=radar)
                if len(pitch_ref):
                    radar = draw_points_on_pitch(CONFIG, pitch_ref, face_color=sv.Color.from_hex("#FFD700"),
                                                 edge_color=sv.Color.BLACK, radius=16, pitch=radar)
                if len(pitch_ball):
                    radar = draw_points_on_pitch(CONFIG, pitch_ball, face_color=sv.Color.WHITE,
                                                 edge_color=sv.Color.BLACK, radius=10, pitch=radar)
                t0, t1 = allp[all_cls == 0], allp[all_cls == 1]
                if len(t0) and len(t1):
                    vor = draw_pitch_voronoi_diagram(CONFIG, t0, t1, sv.Color.from_hex("#00BFFF"),
                                                     sv.Color.from_hex("#FF1493"), pitch=vor)
            sink_radar.write_frame(radar)
            sink_vor.write_frame(vor)
            if i in stills:
                cv2.imwrite(os.path.join(OUT_DIR, f"still_{i}.jpg"), out)
                cv2.imwrite(os.path.join(OUT_DIR, f"radar_{i}.jpg"), radar)
                cv2.imwrite(os.path.join(OUT_DIR, f"voronoi_{i}.jpg"), vor)
            if i % 50 == 0:
                print(f"  frame {i}/{total}  keypoints={n_kp}", flush=True)

    # --- ball path: drop multi-ball frames, then drop jumps over 5 m ---
    raw = [p if len(p) == 1 else np.array([], dtype=np.float64) for p in ball_path]
    raw = [p[0] if len(p) else np.array([], dtype=np.float64) for p in raw]
    cleaned = replace_outliers_based_on_distance(raw, BALL_MAX_JUMP_CM)
    raw_pts = np.array([p for p in raw if len(p)])
    clean_pts = np.array([p for p in cleaned if len(p)])
    if len(raw_pts):
        cv2.imwrite(os.path.join(OUT_DIR, "ball_path_raw.jpg"), draw_paths_on_pitch(CONFIG, [raw_pts]))
    if len(clean_pts):
        cv2.imwrite(os.path.join(OUT_DIR, "ball_path_clean.jpg"), draw_paths_on_pitch(CONFIG, [clean_pts]))

    rows = [json.loads(l) for l in open(os.path.join(OUT_DIR, "frames.jsonl"))]
    summary = {
        "frames": len(rows),
        "distinct_tracker_ids_total": len(ids_seen),
        "mean_players_per_frame": float(np.mean([len(r["players"]) for r in rows])),
        "frames_with_homography_ok_pct": float(np.mean([k >= 4 for k in n_kp_hist]) * 100),
        "mean_keypoints_per_frame": float(np.mean(n_kp_hist)),
        "ball_frames_raw": int(len(raw_pts)),
        "ball_frames_after_cleanup": int(len(clean_pts)),
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
