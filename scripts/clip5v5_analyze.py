#!/usr/bin/env python3
"""
Stage 2 for the fixed-camera 5-a-side clip (reads cache.jsonl from clip5v5_detect.py).

  - undo the small camera pan (dx), keep only people standing on the pitch polygon
  - ByteTrack, then stitch broken tracks and keep a closed roster:
        white team 4 outfield, dark team 4 outfield, 2 keepers (one at each goal)
  - ball: drop static false positives, best path through candidates, interpolate short gaps
  - possession -> passes / turnovers, per-player and per-team metrics, heatmaps, annotated video

Distances/speeds are ESTIMATES: metres-per-pixel come from player height (1.75 m) and an
assumed camera elevation (ELEV_DEG); no pitch calibration exists for this clip.

Usage: python clip5v5_analyze.py <out_dir> [white_defends=left|right]
"""
import sys, json
from pathlib import Path
import numpy as np
import cv2
import supervision as sv

OUT = Path(sys.argv[1])
WHITE_DEFENDS = sys.argv[2] if len(sys.argv) > 2 else None
meta = json.load(open(OUT / "meta.json"))
FPS, STRIDE, VIDEO = meta["fps"], meta["stride"], meta["video"]
SFPS = FPS / STRIDE
POLY = np.array([(95, 195), (350, 188), (600, 175), (820, 220), (1070, 270), (1297, 315),
                 (1297, 730), (165, 730)], np.int32)          # pitch, frame-0 coordinates
PERSON_CONF, BALL_CONF, MIN_H = 0.25, 0.08, 25
H_M, ELEV_DEG = 1.75, 30.0
POSSESS_RATIO, MIN_HOLD, MAX_GAP_S = 0.8, 2, 2.0
MIN_TRACK = 8


def load():
    src = OUT / "cache2.jsonl" if (OUT / "cache2.jsonl").exists() else OUT / "cache.jsonl"
    print("reading", src.name)
    rows = [json.loads(l) for l in open(src)]
    rows.sort(key=lambda r: r["frame"])
    return rows


def in_pitch(x, y):
    return cv2.pointPolygonTest(POLY, (float(x), float(y)), True) >= -10


# ---------------------------------------------------------------- tracking
def track(rows):
    tr = sv.ByteTrack(track_activation_threshold=0.30, lost_track_buffer=int(2 * SFPS),
                      minimum_matching_threshold=0.8, frame_rate=max(1, int(SFPS)))
    per_frame = []
    for r in rows:
        dx = r["dx"]
        P = [d for d in r["dets"] if d[0] != 0 and d[5] >= PERSON_CONF]
        keep = []
        for d in P:
            x1, y1, x2, y2 = d[1] - dx, d[2], d[3] - dx, d[4]
            if y2 - y1 >= MIN_H and in_pitch((x1 + x2) / 2, y2):
                keep.append([x1, y1, x2, y2, d[5], d[0], d[6], d[7], d[8]])
        if keep:
            a = np.array(keep, dtype=np.float32)
            det = sv.Detections(xyxy=a[:, :4], confidence=a[:, 4], class_id=np.zeros(len(a), int))
            det = det.with_nms(threshold=0.5, class_agnostic=True)
            # map boxes back to extra info (colour, class) by nearest centre
            upd = tr.update_with_detections(det)
        else:
            a = np.zeros((0, 9), np.float32)
            upd = tr.update_with_detections(sv.Detections.empty())
        out = []
        for tid, box, cf in zip(upd.tracker_id, upd.xyxy, upd.confidence):
            j = int(np.argmin(np.abs(a[:, :4] - box).sum(1))) if len(a) else None
            out.append(dict(tid=int(tid), box=[float(v) for v in box], conf=float(cf),
                            cls=int(a[j, 5]) if j is not None else 2,
                            rgb=[float(v) for v in a[j, 6:9]] if j is not None else [0, 0, 0]))
        per_frame.append(out)
    return per_frame


def stitch_and_roster(per_frame):
    tracks = {}
    for fi, out in enumerate(per_frame):
        for o in out:
            tracks.setdefault(o["tid"], []).append((fi, o))
    info = {}
    for tid, seq in tracks.items():
        if len(seq) < MIN_TRACK:
            continue
        bright = float(np.median([np.mean(o["rgb"]) for _, o in seq]))
        feet = np.array([[(o["box"][0] + o["box"][2]) / 2, o["box"][3]] for _, o in seq])
        info[tid] = dict(tid=tid, seq=seq, start=seq[0][0], end=seq[-1][0], bright=bright,
                         first=feet[0], last=feet[-1], n=len(seq),
                         gkvote=float(np.mean([o["cls"] == 1 for _, o in seq])))
    # team by brightness (2-means)
    ids = list(info)
    b = np.array([info[i]["bright"] for i in ids], np.float32).reshape(-1, 1)
    _, lab, cen = cv2.kmeans(b, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3), 5,
                             cv2.KMEANS_PP_CENTERS)
    white_label = int(np.argmax(cen.ravel()))
    for i, l in zip(ids, lab.ravel()):
        info[i]["team"] = "W" if int(l) == white_label else "B"
    # stitch: join a track's end to a later track's start (same team, close in space/time)
    chains = {i: [i] for i in ids}
    nxt, prv = {}, {}
    cands = []
    for a in ids:
        for c in ids:
            if a == c or info[c]["team"] != info[a]["team"]:
                continue
            gap = info[c]["start"] - info[a]["end"]
            if 1 <= gap <= int(2.5 * SFPS):
                dist = float(np.linalg.norm(info[c]["first"] - info[a]["last"]))
                if dist <= 80 + 12 * gap:
                    cands.append((gap + dist / 40, a, c))
    for _, a, c in sorted(cands):
        if a in nxt or c in prv:
            continue
        nxt[a], prv[c] = c, a
    idents = []
    for i in ids:
        if i in prv:
            continue
        chain, cur = [i], i
        while cur in nxt:
            cur = nxt[cur]
            chain.append(cur)
        idents.append(chain)
    # per identity summary
    ident = []
    for ch in idents:
        frames = {}
        for tid in ch:
            for fi, o in info[tid]["seq"]:
                frames[fi] = o
        feet = np.array([[(o["box"][0] + o["box"][2]) / 2, o["box"][3]] for o in frames.values()])
        ident.append(dict(tids=ch, frames=frames, team=info[ch[0]]["team"], n=len(frames),
                          mean_x=float(feet[:, 0].mean()), spread=float(feet.std(0).sum()),
                          gkvote=float(np.mean([info[t]["gkvote"] for t in ch]))))
    # closed roster: 4 white outfield, 4 dark outfield, 2 keepers (extreme-x dark-clothed)
    whites = sorted([x for x in ident if x["team"] == "W"], key=lambda x: -x["n"])
    darks = sorted([x for x in ident if x["team"] == "B"], key=lambda x: -x["n"])
    roster = []
    gk_pool = sorted(darks[:6] + whites[4:5], key=lambda x: x["mean_x"])
    keepers = []
    if len(gk_pool) >= 2:
        left, right = gk_pool[0], gk_pool[-1]
        keepers = [left, right]
    outfield_w = [x for x in whites if x not in keepers][:4]
    outfield_b = [x for x in darks if x not in keepers][:4]
    # assign each keeper to a team: whites defend the goal that is farther from their mean position
    if len(keepers) == 2:
        wmean = np.mean([x["mean_x"] for x in outfield_w]) if outfield_w else None
        left_is_white = (WHITE_DEFENDS == "left") if WHITE_DEFENDS else (wmean is not None and wmean < (keepers[0]["mean_x"] + keepers[1]["mean_x"]) / 2)
        keepers[0]["team"], keepers[1]["team"] = ("W", "B") if left_is_white else ("B", "W")
    for x in keepers:
        x["role"] = "GK"
    for x in outfield_w + outfield_b:
        x["role"] = "P"
    roster = keepers + outfield_w + outfield_b
    for team in ("W", "B"):
        for k, x in enumerate([r for r in roster if r["team"] == team and r["role"] == "P"] +
                              [r for r in roster if r["team"] == team and r["role"] == "GK"], 1):
            x["label"] = f"{team}{k}" + ("-GK" if x["role"] == "GK" else "")
    return roster, info, ident


# ------------------------------------------------------------------- ball
def ball_track(rows):
    cands = []
    for fi, r in enumerate(rows):
        for d in r["dets"]:
            if d[0] == 0 and d[5] >= BALL_CONF:
                cx, cy = (d[1] + d[3]) / 2 - r["dx"], (d[2] + d[4]) / 2
                if cy > 120:
                    cands.append((fi, cx, cy, d[5]))
    # static false positives: same 14-px cell in >=20% of frames
    cells = {}
    for fi, cx, cy, _ in cands:
        cells.setdefault((int(cx // 14), int(cy // 14)), set()).add(fi)
    static = {c for c, s in cells.items() if len(s) >= 0.20 * len(rows)}
    cands = [c for c in cands if (int(c[1] // 14), int(c[2] // 14)) not in static]
    # best path: maximise sum(conf+0.5) - sum(dist/60 + 0.5*(gap-1))
    n = len(cands)
    best, back = [0.0] * n, [-1] * n
    for j in range(n):
        best[j] = cands[j][3] + 0.5
        for i in range(j - 1, -1, -1):
            gap = cands[j][0] - cands[i][0]
            if gap < 1:
                continue
            if gap > 6:
                break
            dist = float(np.hypot(cands[j][1] - cands[i][1], cands[j][2] - cands[i][2]))
            if dist > 110 * gap:
                continue
            s = best[i] + cands[j][3] + 0.5 - dist / 60 - 0.5 * (gap - 1)
            if s > best[j]:
                best[j], back[j] = s, i
    path = []
    if n:
        j = int(np.argmax(best))
        while j != -1:
            path.append(cands[j])
            j = back[j]
        path.reverse()
    ball = {}
    for (f0, x0, y0, _), (f1, x1, y1, _) in zip(path, path[1:]):
        for f in range(f0, f1 + 1):
            t = (f - f0) / (f1 - f0)
            ball[f] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0), f in (f0, f1))
    if path:
        ball[path[-1][0]] = (path[-1][1], path[-1][2], True)
    return ball, len(cands), len(path), len(static)


# --------------------------------------------------------- passes + metrics
def metres(dx, dy, h):
    th = np.radians(ELEV_DEG)
    return dx * H_M * np.cos(th) / h, dy * H_M / (h * np.tan(th))


def analyse(rows, roster, ball):
    nF = len(rows)
    holder = []
    for fi in range(nF):
        b = ball.get(fi)
        best = None
        if b:
            for k, x in enumerate(roster):
                o = x["frames"].get(fi)
                if not o:
                    continue
                bx1, by1, bx2, by2 = o["box"]
                h = by2 - by1
                d = np.hypot(b[0] - (bx1 + bx2) / 2, b[1] - (by2 - 0.1 * h)) / h
                if d < POSSESS_RATIO and (best is None or d < best[0]):
                    best = (d, k)
        holder.append(best[1] if best else None)
    runs, cur, start = [], None, 0
    for i, h in enumerate(holder + [None]):
        if h != cur:
            if cur is not None and i - start >= MIN_HOLD:
                runs.append([cur, start, i - 1])
            cur, start = h, i
    merged = []
    for r in runs:                     # merge same-holder runs separated by short gaps
        if merged and merged[-1][0] == r[0] and r[1] - merged[-1][2] <= 3:
            merged[-1][2] = r[2]
        else:
            merged.append(r)
    events = []
    for a, b in zip(merged, merged[1:]):
        if a[0] == b[0]:
            continue
        gap = (b[1] - a[2]) / SFPS
        if gap > MAX_GAP_S:
            continue
        pa, pb = ball.get(a[2]), ball.get(b[1])
        h = np.mean([roster[a[0]]["frames"][a[2]]["box"][3] - roster[a[0]]["frames"][a[2]]["box"][1]])
        travel = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])) if pa and pb else 0
        if travel < 0.6 * h:
            continue
        same = roster[a[0]]["team"] == roster[b[0]]["team"]
        events.append(dict(frame=b[1], t=round((rows[b[1]]["frame"] - rows[0]["frame"]) / FPS, 2),
                           kind="pass" if same else "turnover", team=roster[a[0]]["team"],
                           frm=roster[a[0]]["label"], to=roster[b[0]]["label"], gap_s=round(gap, 2),
                           ball_travel_px=round(travel)))
    # player metrics
    pm = {}
    for k, x in enumerate(roster):
        fr = sorted(x["frames"])
        pts = np.array([[(x["frames"][f]["box"][0] + x["frames"][f]["box"][2]) / 2, x["frames"][f]["box"][3],
                         x["frames"][f]["box"][3] - x["frames"][f]["box"][1]] for f in fr])
        if len(pts) >= 5:                                    # smooth
            ker = np.ones(5) / 5
            pts[:, 0] = np.convolve(np.pad(pts[:, 0], 2, mode="edge"), ker, "valid")
            pts[:, 1] = np.convolve(np.pad(pts[:, 1], 2, mode="edge"), ker, "valid")
        dist, speeds = 0.0, []
        for i in range(1, len(pts)):
            dt = (fr[i] - fr[i - 1]) / SFPS
            if dt <= 0 or dt > 1.0:
                continue
            h = (pts[i, 2] + pts[i - 1, 2]) / 2
            mx, my = metres(pts[i, 0] - pts[i - 1, 0], pts[i, 1] - pts[i - 1, 1], h)
            step = float(np.hypot(mx, my))
            if step / dt > 9:                                # implausible -> id/box jump
                continue
            dist += step
            speeds.append(step / dt)
        sp = np.convolve(speeds, np.ones(3) / 3, "valid") if len(speeds) >= 3 else np.array(speeds)
        mine = [r for r in merged if r[0] == k]
        pm[x["label"]] = dict(
            team=x["team"], role=x["role"], track_ids=x["tids"],
            seen_pct=round(100 * len(fr) / nF), distance_m=round(dist, 1),
            avg_speed_ms=round(float(np.mean(speeds)), 2) if speeds else None,
            top_speed_ms=round(float(sp.max()), 2) if len(sp) else None,
            possessions=len(mine), time_on_ball_s=round(sum(r[2] - r[1] + 1 for r in mine) / SFPS, 2),
            passes_made=sum(e["kind"] == "pass" and e["frm"] == x["label"] for e in events),
            passes_received=sum(e["kind"] == "pass" and e["to"] == x["label"] for e in events),
            turnovers=sum(e["kind"] == "turnover" and e["frm"] == x["label"] for e in events),
            interceptions=sum(e["kind"] == "turnover" and e["to"] == x["label"] for e in events),
            mean_pos_px=[round(float(pts[:, 0].mean())), round(float(pts[:, 1].mean()))])
    # team metrics
    tm = {}
    held = {"W": 0, "B": 0}
    for r in merged:
        held[roster[r[0]]["team"]] += r[2] - r[1] + 1
    tot = max(sum(held.values()), 1)
    for t in ("W", "B"):
        outf = [x for x in roster if x["team"] == t and x["role"] == "P"]
        spreads, cents = [], []
        for fi in range(nF):
            ps = [((x["frames"][fi]["box"][0] + x["frames"][fi]["box"][2]) / 2, x["frames"][fi]["box"][3])
                  for x in outf if fi in x["frames"]]
            if len(ps) >= 3:
                a = np.array(ps)
                cents.append(a.mean(0))
                spreads.append((np.ptp(a[:, 0]), np.ptp(a[:, 1])))
        tm[t] = dict(
            possession_pct=round(100 * held[t] / tot), passes=sum(e["kind"] == "pass" and e["team"] == t for e in events),
            turnovers=sum(e["kind"] == "turnover" and e["team"] == t for e in events),
            pass_accuracy_pct=(round(100 * sum(e["kind"] == "pass" and e["team"] == t for e in events) /
                                     max(sum(e["team"] == t for e in events), 1))),
            mean_centroid_px=[round(float(np.mean([c[0] for c in cents]))), round(float(np.mean([c[1] for c in cents])))] if cents else None,
            mean_width_px=round(float(np.mean([s[0] for s in spreads]))) if spreads else None,
            mean_depth_px=round(float(np.mean([s[1] for s in spreads]))) if spreads else None,
            total_distance_m=round(sum(v["distance_m"] for v in pm.values() if v["team"] == t), 1))
    # ball
    bpts = sorted(ball.items())
    bdist = 0.0
    bspeeds = []
    for (f0, a), (f1, b) in zip(bpts, bpts[1:]):
        if f1 - f0 != 1:
            continue
        near = [x["frames"][f1]["box"][3] - x["frames"][f1]["box"][1] for x in roster if f1 in x["frames"]]
        h = float(np.median(near)) if near else 100
        mx, my = metres(b[0] - a[0], b[1] - a[1], h)
        step = float(np.hypot(mx, my))
        bdist += step
        bspeeds.append(step * SFPS)
    ballm = dict(frames_with_ball=len(ball), pct_of_frames=round(100 * len(ball) / nF),
                 real_detections=sum(1 for v in ball.values() if v[2]),
                 est_distance_m=round(bdist, 1),
                 est_avg_speed_ms=round(float(np.mean(bspeeds)), 1) if bspeeds else None,
                 est_max_speed_ms=round(float(np.percentile(bspeeds, 95)), 1) if bspeeds else None)
    return events, pm, tm, ballm, holder, merged


# ------------------------------------------------------------------- render
def render(rows, roster, ball, events, merged, holder, pm):
    cap = cv2.VideoCapture(VIDEO)
    w = cv2.VideoWriter(str(OUT / "check.webm"), cv2.VideoWriter_fourcc(*"VP80"), SFPS, (1280, 720))
    col = {"W": (255, 220, 0), "B": (0, 140, 255)}
    trail = []
    heat = {"W": np.zeros((720, 1280), np.float32), "B": np.zeros((720, 1280), np.float32)}
    first = None
    cap2 = cv2.VideoCapture(VIDEO)
    for fi, r in enumerate(rows):
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, frame = cap.read()
        if not ok:
            break
        if first is None:
            first = frame.copy()
        dx = int(round(r["dx"]))
        cv2.polylines(frame, [POLY + np.array([dx, 0])], True, (80, 80, 80), 1)
        for k, x in enumerate(roster):
            o = x["frames"].get(fi)
            if not o:
                continue
            x1, y1, x2, y2 = [int(v) for v in o["box"]]
            x1, x2 = x1 + dx, x2 + dx
            c = col[x["team"]]
            thick = 4 if holder[fi] == k else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, thick)
            cv2.putText(frame, x["label"], (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
            heat[x["team"]][min(int(y2), 719), min(max(int((x1 + x2) / 2 - dx), 0), 1279)] += 1
        b = ball.get(fi)
        if b:
            trail.append((int(b[0] + dx), int(b[1])))
            trail = trail[-12:]
            cv2.circle(frame, trail[-1], 12, (0, 0, 255), 3)
        for p, q in zip(trail, trail[1:]):
            cv2.line(frame, p, q, (0, 0, 255), 2)
        for e in [e for e in events if 0 <= (fi - e["frame"]) / SFPS <= 2.0]:
            cv2.putText(frame, f"{e['kind'].upper()} {e['frm']} -> {e['to']}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (0, 255, 255), 3)
        wp = sum(e["kind"] == "pass" and e["team"] == "W" and e["frame"] <= fi for e in events)
        bp = sum(e["kind"] == "pass" and e["team"] == "B" and e["frame"] <= fi for e in events)
        cv2.putText(frame, f"t={(r['frame'] - rows[0]['frame']) / FPS:4.1f}s  passes W:{wp} B:{bp}", (30, 700),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        w.write(frame)
    w.release()
    for t, nm in (("W", "white"), ("B", "dark")):
        h = cv2.GaussianBlur(heat[t], (0, 0), 25)
        if h.max() > 0:
            h = (255 * h / h.max()).astype(np.uint8)
            img = cv2.addWeighted(first, 0.55, cv2.applyColorMap(h, cv2.COLORMAP_JET), 0.45, 0)
            cv2.imwrite(str(OUT / f"heatmap_{nm}.jpg"), img)


def main():
    rows = load()
    per_frame = track(rows)
    roster, info, ident = stitch_and_roster(per_frame)
    for x in roster:
        pass
    ball, ncand, npath, nstatic = ball_track(rows)
    events, pm, tm, ballm, holder, merged = analyse(rows, roster, ball)
    # attach tracked frames to roster entries in frame-index space (already)
    render(rows, roster, ball, events, merged, holder, pm)
    summary = dict(
        clip=VIDEO, seconds=round((rows[-1]["frame"] - rows[0]["frame"]) / FPS + STRIDE / FPS, 1),
        sampled_frames=len(rows), sampled_fps=round(SFPS, 1),
        raw_tracker_ids=len({o["tid"] for f in per_frame for o in f}),
        tracks_after_min_length=len(info), identities_after_stitching=len(ident),
        roster=[dict(label=x["label"], team=x["team"], role=x["role"], tids=x["tids"], frames=x["n"]) for x in roster],
        white_defends=WHITE_DEFENDS or "guessed from positions (check this)",
        ball=dict(candidates_after_static_filter=ncand, static_spots_removed=nstatic, path_nodes=npath, **ballm),
        passes_total=sum(e["kind"] == "pass" for e in events),
        turnovers_total=sum(e["kind"] == "turnover" for e in events),
        events=events, teams=tm, players=pm,
        assumptions=dict(player_height_m=H_M, camera_elevation_deg=ELEV_DEG,
                         note="distances and speeds are rough estimates, no pitch calibration"))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("events", "players", "roster")}, indent=1))
    print("events:", json.dumps(events))
    for k, v in pm.items():
        print(k, v)


if __name__ == "__main__":
    main()
