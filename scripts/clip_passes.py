#!/usr/bin/env python3
"""
Basic passes for a fixed-camera clip: identities (clip_ids.py) + ball candidates (clip_ball.py).

  python clip_passes.py <video> <out_dir> <start_s> <seconds> <n_players>

Teams: the identities are ranked by torso brightness; the lighter half is team W, the darker half team B
(Tolworth: white vs black kits). Ball: best path through the candidates with a strict jump limit.
Possession: ball within 0.8 player-heights of a player's feet AND a real detection within 3 sampled
frames (an interpolated ball does not give possession). A pass = possession passes to a different
player of the same team within 2 s with the ball having travelled >= 0.6 player-heights; to a player
of the other team = turnover.

Writes passes.json, passes.webm and pass_sheet.jpg (one still per event, for a quick eyeball check).
"""
import sys, os, json
from pathlib import Path
import numpy as np
import cv2

video, OUT, start_s, seconds, n = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
sys.argv = ["clip_ids.py", video, str(OUT), start_s, seconds, n]
sys.path.insert(0, str(Path(__file__).parent))
import clip_ids as C                                   # noqa: E402

BALL_CONF = 0.08
JUMP_PX_PER_ROW = 50      # max ball movement between candidates, per sampled frame of gap
POSSESS_RATIO, MIN_HOLD, MAX_GAP_S, MIN_TRAVEL = 0.8, 2, 2.0, 0.6
SUPPORT_ROWS = 3
SCALE = 1.0               # frame width / 1280; pixel thresholds below are tuned at 720p


def ball_path(rows, ballrows, nrows):
    cands = []
    for fi, cs in ballrows.items():
        for c in cs:
            if c[4] >= BALL_CONF and (c[1] + c[3]) / 2 > 100 * SCALE:
                cands.append((fi, (c[0] + c[2]) / 2, (c[1] + c[3]) / 2, c[4]))
    cells = {}
    for fi, x, y, _ in cands:                      # drop spots that never move (markings, bags, goal parts)
        cells.setdefault((int(x // (14 * SCALE)), int(y // (14 * SCALE))), set()).add(fi)
    static = {k for k, s in cells.items() if len(s) >= 0.15 * nrows}
    cands = sorted([c for c in cands if (int(c[1] // (14 * SCALE)), int(c[2] // (14 * SCALE))) not in static])
    det_rows = sorted(ballrows)
    STEP = max(1, int(np.median(np.diff(det_rows)))) if len(det_rows) > 1 else 1      # rows between ball detections
    m = len(cands)
    best, back = [0.0] * m, [-1] * m
    for j in range(m):
        best[j] = cands[j][3] + 0.5
        for i in range(j - 1, -1, -1):
            gap = cands[j][0] - cands[i][0]
            if gap < 1:
                continue
            if gap > 8 * STEP:
                break
            dist = float(np.hypot(cands[j][1] - cands[i][1], cands[j][2] - cands[i][2]))
            if dist > JUMP_PX_PER_ROW * SCALE * gap:
                continue
            s = best[i] + cands[j][3] + 0.5 - dist / (40 * SCALE) - 0.4 * max(0.0, gap / STEP - 1)
            if s > best[j]:
                best[j], back[j] = s, i
    path = []
    if m:
        j = int(np.argmax(best))
        while j != -1:
            path.append(cands[j])
            j = back[j]
        path.reverse()
    ball, real = {}, [p[0] for p in path]
    for (f0, x0, y0, _), (f1, x1, y1, _) in zip(path, path[1:]):
        for f in range(f0, f1 + 1):
            t = (f - f0) / (f1 - f0)
            ball[f] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
    if path:
        ball[path[-1][0]] = (path[-1][1], path[-1][2])
    return ball, np.array(real), len(cands), len(path), len(static)


def main():
    global SCALE
    meta = json.load(open(OUT / "meta.json"))
    SCALE = meta["width"] / 1280.0
    rows = sorted((json.loads(l) for l in open(OUT / "dets.jsonl")), key=lambda r: r["frame"])
    idx_of = {r["frame"]: i for i, r in enumerate(rows)}
    ballrows = {}
    for name in ("ball.jsonl", "ball_motion.jsonl"):          # neural ball models + motion blobs, merged per frame
        if (OUT / name).exists():
            for l in open(OUT / name):
                j = json.loads(l)
                ballrows.setdefault(idx_of[j["frame"]], []).extend(j["cands"])
    ballrows = {k: v for k, v in ballrows.items()}
    nrows_for_static = len(rows)
    sfps = meta["fps"] / meta["stride"]
    per_frame, _ = C.track(rows, np.ones((meta["height"], meta["width"]), np.uint8), [], sfps)
    colourer = C.TrackletColour(video, rows) if os.environ.get("COLOUR_LOCK") else None
    ident, unassigned, *_ = C.lock_identities(per_frame, sfps, colourer=colourer)
    cols = {}
    if os.environ.get("RETEAM"):                    # teams from grass-free shirt colour (3 groups: two teams + odd colours)
        cols = C.reteam(meta["video"], rows, ident)
    officials = set()
    for tm in ("W", "B"):                           # a shirt colour far from the rest of its team (e.g. a referee in black among blue shirts)
        mem = [k for k, I in enumerate(ident) if I.get("team") == tm and k in cols]
        if len(mem) >= 5:
            med = np.median([cols[k] for k in mem], axis=0)
            d = {k: float(np.linalg.norm(cols[k] - med)) for k in mem}
            thr = max(3 * float(np.median(list(d.values()))), 25.0)
            for k in mem:
                if d[k] > thr:
                    officials.add(k); print(f"colour outlier in team {tm}: identity {k} (distance {d[k]:.0f}, limit {thr:.0f}) -> treated as an official", flush=True)
    if os.environ.get("HFILE"):                     # officials = people who spend most of the time on/outside the pitch lines
        Hinv = np.linalg.inv(np.load(os.environ["HFILE"]))
        PL, PW = float(os.environ.get("PITCH_L", 105)), float(os.environ.get("PITCH_W", 68))
        for k, I in enumerate(ident):
            pts = []
            for o in I["frames"].values():
                p_ = Hinv @ np.array([(o["box"][0] + o["box"][2]) / 2, o["box"][3], 1.0]); pts.append((p_[0] / p_[2], p_[1] / p_[2]))
            X, Y = np.array(pts).T
            if np.mean((Y < 1.5) | (Y > PW - 1.5) | (X < 1.5) | (X > PL - 1.5)) >= 0.7:
                officials.add(k)
    team = {}
    gk_side = {}
    for k, I in enumerate(ident):
        team[k] = "O" if k in officials else I.get("team", "?") if I.get("team") in ("W", "B") else "?"
    if os.environ.get("HFILE"):                     # odd-colour people: near a goal = keeper, elsewhere = referee
        for k, I in enumerate(ident):
            if team[k] != "?":
                continue
            xs_ = [(lambda p_: p_[0] / p_[2])(Hinv @ np.array([(o["box"][0] + o["box"][2]) / 2, o["box"][3], 1.0])) for o in I["frames"].values()]
            mx = float(np.mean(xs_))
            if mx < 22 or mx > PL - 22:
                gk_side[k] = "L" if mx < PL / 2 else "R"
                forced = os.environ.get("GK_LEFT_TEAM" if gk_side[k] == "L" else "GK_RIGHT_TEAM")   # e.g. GK_LEFT_TEAM=W
                if forced in ("W", "B"):
                    team[k] = forced
            else:
                team[k] = "O"
    counter = {"W": 0, "B": 0, "?": 0, "O": 0}
    label = {}
    for k in sorted(team, key=lambda k: (team[k], k)):
        counter[team[k]] += 1
        label[k] = (f"{team[k]}{counter[team[k]]}" + ("-GK" if k in gk_side else "")) if team[k] in ("W", "B") else ("REF" if team[k] == "O" else f"GK-{gk_side.get(k, '?')}")
    print("identities by team:", {t: sum(1 for v in team.values() if v == t) for t in ("W", "B", "?", "O")}, flush=True)

    ball, real, ncand, npath, nstatic = ball_path(rows, ballrows, len(rows))
    holder = []
    for fi in range(len(rows)):
        b = ball.get(fi)
        best = None
        if b is not None and len(real) and np.min(np.abs(real - fi)) <= SUPPORT_ROWS:
            for k, I in enumerate(ident):
                o = I["frames"].get(fi)
                if not o:
                    continue
                x1, y1, x2, y2 = o["box"]
                h = y2 - y1
                d = np.hypot(b[0] - (x1 + x2) / 2, b[1] - (y2 - 0.1 * h)) / h
                if d < POSSESS_RATIO and (best is None or d < best[0]):
                    best = (d, k)
        holder.append(best[1] if best else None)
    runs, cur, st = [], None, 0
    for i, h in enumerate(holder + [None]):
        if h != cur:
            if cur is not None and i - st >= MIN_HOLD:
                runs.append([cur, st, i - 1])
            cur, st = h, i
    merged = []
    for r in runs:
        if merged and merged[-1][0] == r[0] and r[1] - merged[-1][2] <= 3:
            merged[-1][2] = r[2]
        else:
            merged.append(r)
    events = []
    for a, b in zip(merged, merged[1:]):
        if a[0] == b[0]:
            continue
        gap = (b[1] - a[2]) / sfps
        if gap > MAX_GAP_S:
            continue
        pa, pb = ball.get(a[2]), ball.get(b[1])
        o = ident[a[0]]["frames"][a[2]]
        h = o["box"][3] - o["box"][1]
        travel = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])) if pa and pb else 0
        if travel < MIN_TRAVEL * h:
            continue
        events.append(dict(row=b[1], t=round(float(start_s) + b[1] / sfps, 2),
                           kind=("pass" if team[a[0]] == team[b[0]] else "turnover") if (team[a[0]] in ("W", "B") and team[b[0]] in ("W", "B")) else "other", team=team[a[0]],
                           frm=label[a[0]], to=label[b[0]], gap_s=round(gap, 2), ball_travel_px=round(travel),
                           frm_row=a[2]))
    held = {"W": 0, "B": 0, "?": 0, "O": 0}
    for r in merged:
        held[team[r[0]]] += r[2] - r[1] + 1
    tot = max(sum(held.values()), 1)
    teams = {t: dict(possession_pct=round(100 * held[t] / tot),
                     passes=sum(e["kind"] == "pass" and e["team"] == t for e in events),
                     turnovers=sum(e["kind"] == "turnover" and e["team"] == t for e in events)) for t in ("W", "B")}
    players = {label[k]: dict(identity=k + 1, seen_pct=round(100 * len(I["frames"]) / len(rows)),
                              possessions=sum(1 for r in merged if r[0] == k),
                              time_on_ball_s=round(sum(r[2] - r[1] + 1 for r in merged if r[0] == k) / sfps, 1),
                              passes_made=sum(e["kind"] == "pass" and e["frm"] == label[k] for e in events),
                              passes_received=sum(e["kind"] == "pass" and e["to"] == label[k] for e in events))
               for k, I in enumerate(ident)}
    summary = dict(seconds=float(seconds), team_by_identity={str(k + 1): label[k] for k in team},
                   ball=dict(candidates=ncand, path_points=npath, static_spots_removed=nstatic,
                             rows_with_ball_pct=round(100 * len(ball) / len(rows)),
                             rows_with_possession_pct=round(100 * sum(h is not None for h in holder) / len(rows))),
                   events=events, teams=teams, players=players)
    (OUT / "passes.json").write_text(json.dumps(summary, indent=1))
    (OUT / "identities.json").write_text(json.dumps({str(k): dict(
        label=label[k], team=team[k], role=("GK" if k in gk_side else ("REF" if team[k] == "O" else "P")),
        rows={str(f): [round(v, 1) for v in o["box"]] + [int(bool(o.get("interp")))] for f, o in I["frames"].items()})
        for k, I in enumerate(ident)}))
    print(json.dumps({k: v for k, v in summary.items() if k != "players"}, indent=1))

    if os.environ.get("FAST") and not os.environ.get("STILLS"):
        return
    # video + one still per event
    cap = cv2.VideoCapture(video)
    pano = meta["width"] / meta["height"] > 2.5
    y0 = int(meta.get("band_y0", 0)) if pano else 0
    xo, xw, y1 = 0, meta["width"], meta["height"]
    if os.environ.get("VIEW"):                      # VIEW="x0,x1,y0,y1,scale": render only that window of the frame
        xo, xw, y0, y1, sc = [float(v) for v in os.environ["VIEW"].split(",")]
        xo, xw, y0, y1 = int(xo), int(xw), int(y0), int(y1)
        W, H = int((xw - xo) * sc) // 2 * 2, int((y1 - y0) * sc) // 2 * 2
    else:
        W = 3200 if pano else 1280
        H = int(W * (meta["height"] - y0) / meta["width"])
        sc = W / meta["width"]
    wr = cv2.VideoWriter(str(OUT / "passes.webm"), cv2.VideoWriter_fourcc(*"VP80"), sfps, (W, H))
    col = {"W": (255, 255, 255), "B": (255, 140, 30), "?": (0, 255, 0), "O": (0, 0, 255)}    # BGR: white team = white boxes, blue team = BLUE boxes
    owner = {}
    for k, I in enumerate(ident):
        for f in I["frames"]:
            owner[(k, f)] = True
    stills = {}
    trail = []
    cur = 0
    for fi, r in enumerate(rows):
        if os.environ.get("STILLS") and not any(e["row"] == fi for e in events):
            continue
        if os.environ.get("STILLS") or fi == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"]); cur = r["frame"]
        while cur < r["frame"]:                      # read in order: seeking in a huge video is very slow
            cap.grab(); cur += 1
        ok, frame = cap.read(); cur += 1
        if not ok:
            break
        full = frame
        frame = cv2.resize(frame[y0:y1, xo:xw], (W, H), interpolation=cv2.INTER_AREA)
        for k, I in enumerate(ident):
            o = I["frames"].get(fi)
            if o:
                b = [int(v * sc) for v in (o["box"][0] - xo, o["box"][1] - y0, o["box"][2] - xo, o["box"][3] - y0)]
                thick = 4 if holder[fi] == k else 2
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), col[team[k]], thick)
                cv2.putText(frame, label[k], (b[0], max(14, b[1] - 4)), 0, 0.6, col[team[k]], 2)
        b = ball.get(fi)
        if b is not None:
            trail.append((fi, int((b[0] - xo) * sc), int((b[1] - y0) * sc)))
            cv2.circle(frame, (int((b[0] - xo) * sc), int((b[1] - y0) * sc)), 9, (0, 0, 255), 2)
        for (tf0, tx0, ty0), (tf1, tx1, ty1) in zip(trail, trail[1:]):
            if fi - tf1 <= 8 and tf1 - tf0 == 1:
                cv2.line(frame, (tx0, ty0), (tx1, ty1), (0, 0, 255), 1)
        txt = [e for e in events if 0 <= fi - e["row"] <= int(1.5 * sfps)]
        for i, e in enumerate(txt):
            cv2.putText(frame, f"{e['kind'].upper()} {e['frm']} -> {e['to']}", (12, 80 + 34 * i), 0, 1.0,
                        (0, 255, 255), 3)
        cv2.rectangle(frame, (0, 0), (620, 44), (0, 0, 0), -1)
        pw = sum(e["kind"] == "pass" and e["team"] == "W" and e["row"] <= fi for e in events)
        pb = sum(e["kind"] == "pass" and e["team"] == "B" and e["row"] <= fi for e in events)
        cv2.putText(frame, f"t={float(start_s) + fi / sfps:5.1f}s  passes so far: W {pw}  B {pb}",
                    (10, 32), 0, 0.8, (255, 255, 255), 2)
        wr.write(frame)
        for ei, e in enumerate(events):
            if fi == e["row"]:
                if pano and ball.get(fi) is not None:
                    bx, by = int(ball[fi][0]), int(ball[fi][1])
                    x1_ = min(max(0, bx - 500), meta["width"] - 1000)
                    crop = full[max(0, by - 260):max(0, by - 260) + 520, x1_:x1_ + 1000].copy()
                    for k, I in enumerate(ident):
                        o = I["frames"].get(fi)
                        if o:
                            bb = [int(o["box"][0]) - x1_, int(o["box"][1]) - max(0, by - 260), int(o["box"][2]) - x1_, int(o["box"][3]) - max(0, by - 260)]
                            cv2.rectangle(crop, (bb[0], bb[1]), (bb[2], bb[3]), col[team[k]], 4 if holder[fi] == k else 2)
                            cv2.putText(crop, label[k], (bb[0], max(14, bb[1] - 4)), 0, 0.7, col[team[k]], 2)
                    cv2.circle(crop, (bx - x1_, 260 if by >= 260 else by), 14, (0, 0, 255), 2)
                    stills[ei] = crop
                else:
                    stills[ei] = frame.copy()
    wr.release()
    tiles = []
    for ei, e in enumerate(events[:12]):
        if ei in stills:
            im = cv2.resize(stills[ei], (640, int(640 * stills[ei].shape[0] / stills[ei].shape[1])))
            cv2.putText(im, f"{e['kind']} {e['frm']}->{e['to']} t={e['t']}s", (8, 28), 0, 0.8, (0, 255, 255), 2)
            tiles.append(im)
    if tiles:
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(OUT / "pass_sheet.jpg"),
                    np.vstack([np.hstack([cv2.copyMakeBorder(t, 0, max(0, max(x.shape[0] for x in tiles[i:i + 2]) - t.shape[0]), 0, 0, cv2.BORDER_CONSTANT) for t in tiles[i:i + 2]]) for i in range(0, len(tiles), 2)]))
    print("wrote passes.webm / pass_sheet.jpg")


if __name__ == "__main__":
    main()
