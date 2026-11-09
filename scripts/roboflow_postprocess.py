#!/usr/bin/env python3
"""
Post-processing for the exact-tutorial run (roboflow_full_pipeline.py +
roboflow_cache_detections.py):

  1. Swap repair: ByteTrack sometimes swaps IDs between two close opposing
     players. Seen as two tracks whose smoothed team label flips at the same
     moment, within a few metres of each other. Their identities are swapped
     back after the change point. A sustained flip with no nearby opposing
     player is not a swap (nobody to swap with) - it's classifier noise, so it
     is only reported, and the track-level vote overrules it.
  2. Track-level team vote: one team per (repaired) track instead of one per
     frame, so single-frame SigLIP noise can't flip a player.
  3. Goalkeepers: one team per goalkeeper track by nearest-centroid vote.
  4. Homography repair: single-frame keypoint glitches throw the projected
     pitch position by 20-400 m. A matrix whose projection of a few probe
     pixels disagrees with the median of its neighbours is replaced by the
     last good one. Players and ball are then (re)projected with the repaired
     matrices.
  5. Ball: Viterbi over every ball candidate per frame in PIXEL space (the ball
     moves smoothly there even when airborne), then gaps interpolated and the
     result projected with that frame's homography.

Usage: python roboflow_postprocess.py <pipeline_dir>
Needs <pipeline_dir>/frames.jsonl and cache.jsonl. Writes tracks_clean.json,
ball_clean.json and postprocess_summary.json.
"""
import sys
import os
import json
from collections import defaultdict, Counter

import numpy as np

D = sys.argv[1]
SMOOTH = 25  # frames, majority window for the team-label series
MIN_RUN = 60  # a team change must hold this long to count as real
SWAP_FRAMES = 40  # two tracks' change points must be this close in time...
SWAP_CM = 300  # ...and this close on the pitch (3 m)

BALL_MAX_PX_PER_FRAME = 90
BALL_MISS_COST = 1.0
BALL_JUMP_COST = 1.0 / 60.0
BALL_MAX_GAP = 20  # interpolate across at most this many frames
BALL_STATIC_FRAMES = 25  # a candidate sitting within 4 px this long is a marking, not the ball


def load(name):
    return [json.loads(l) for l in open(os.path.join(D, name))]


def change_points(frames, labels):
    """Frames where the smoothed team label starts a run lasting >= MIN_RUN."""
    labels = np.asarray(labels)
    sm = np.array([round(labels[max(0, i - SMOOTH // 2): i + SMOOTH // 2 + 1].mean()) for i in range(len(labels))])
    runs, st = [], 0
    for i in range(1, len(sm) + 1):
        if i == len(sm) or sm[i] != sm[st]:
            runs.append([st, i - 1, int(sm[st])])
            st = i
    # merge short runs into their neighbours
    merged = []
    for r in runs:
        if merged and (r[1] - r[0] + 1 < MIN_RUN or merged[-1][2] == r[2]):
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    if len(merged) > 1 and merged[0][1] - merged[0][0] + 1 < MIN_RUN:
        merged[1][0] = merged[0][0]
        merged = merged[1:]
    return [(frames[r[0]], r[2]) for r in merged[1:]], merged


def repair_tracks(rows):
    # per-track time series
    ser = defaultdict(dict)  # tid -> frame -> (team, xy)
    for r in rows:
        for t, c, xy in r["players"]:
            ser[t][r["frame"]] = (c, xy)
    cps = {}
    for t, s in ser.items():
        fr = sorted(s)
        if len(fr) >= 2 * MIN_RUN:
            pts, _ = change_points(fr, [s[f][0] for f in fr])
            if pts:
                cps[t] = pts
    swaps, unresolved = [], []
    used = set()
    tids = sorted(cps)
    for a in tids:
        for fa, ta in cps[a]:
            if (a, fa) in used:
                continue
            for b in tids:
                if b <= a:
                    continue
                for fb, tb in cps[b]:
                    if (b, fb) in used or abs(fa - fb) > SWAP_FRAMES or ta == tb:
                        continue
                    f = min(fa, fb)
                    if f in ser[a] and f in ser[b]:
                        d = np.linalg.norm(np.array(ser[a][f][1]) - np.array(ser[b][f][1]))
                        if d <= SWAP_CM:
                            swaps.append({"a": a, "b": b, "frame": max(fa, fb), "dist_cm": round(float(d))})
                            used.update({(a, fa), (b, fb)})
    for t in tids:
        for fr_, _ in cps[t]:
            if (t, fr_) not in used:
                unresolved.append({"track": t, "frame": fr_})
    return swaps, unresolved


def apply_swaps(rows, swaps):
    for r in rows:
        f = r["frame"]
        swapmap = {}
        for s in swaps:
            if f >= s["frame"]:
                swapmap[s["a"]], swapmap[s["b"]] = s["b"], s["a"]
        new = []
        for t, c, xy in r["players"]:
            t2 = swapmap.get(t, t)
            new.append([t2, c, xy])
        r["players"] = new
    return rows


def vote_teams(rows):
    votes = defaultdict(list)
    for r in rows:
        for t, c, _ in r["players"]:
            votes[t].append(c)
    team = {t: Counter(v).most_common(1)[0][0] for t, v in votes.items()}
    # goalkeepers: nearest team centroid of the *voted* player teams, per frame, then vote per track
    gvotes = defaultdict(list)
    for r in rows:
        pl = [(team[t], xy) for t, _, xy in r["players"]]
        t0 = [xy for c, xy in pl if c == 0]
        t1 = [xy for c, xy in pl if c == 1]
        if not t0 or not t1:
            continue
        c0, c1 = np.mean(t0, axis=0), np.mean(t1, axis=0)
        for t, _, xy in r["goalkeepers"]:
            gvotes[t].append(0 if np.linalg.norm(np.array(xy) - c0) < np.linalg.norm(np.array(xy) - c1) else 1)
    gteam = {t: Counter(v).most_common(1)[0][0] for t, v in gvotes.items()}
    return team, gteam, votes


def track_ball(cache):
    """Viterbi over candidates per frame; returns [(frame, cx, cy, conf, interpolated)]."""
    cands = []
    for r in cache:
        cands.append([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[4]) for b in r["ball"]])
    # drop static false positives (field markings, boots on the touchline)
    all_pts = [(i, c[0], c[1]) for i, cs in enumerate(cands) for c in cs]
    keep = [[True] * len(cs) for cs in cands]
    by_cell = defaultdict(list)
    for i, x, y in all_pts:
        by_cell[(round(x / 4), round(y / 4))].append(i)
    static_cells = {k for k, v in by_cell.items() if len(set(v)) >= BALL_STATIC_FRAMES}
    for i, cs in enumerate(cands):
        for j, c in enumerate(cs):
            if (round(c[0] / 4), round(c[1] / 4)) in static_cells:
                keep[i][j] = False
    cands = [[c for c, k in zip(cs, ks) if k] for cs, ks in zip(cands, keep)]

    n = len(cands)
    # state per frame: index into cands[i], or -1 for "missing"
    INF = 1e18
    cost = [None] * n
    back = [None] * n
    prev_states = None
    last_seen = [None] * n  # last real position along the best path, per state (for missing runs)
    for i in range(n):
        states = list(range(len(cands[i]))) + [-1]
        c_i = np.full(len(states), INF)
        b_i = np.zeros(len(states), dtype=int)
        pos_i = [None] * len(states)
        for si, s in enumerate(states):
            unary = BALL_MISS_COST if s == -1 else -cands[i][s][2]
            if i == 0:
                c_i[si] = unary
                pos_i[si] = None if s == -1 else cands[i][s][:2]
                continue
            best, bi = INF, 0
            for pi, ps in enumerate(prev_states):
                pc = cost[i - 1][pi]
                if pc >= INF:
                    continue
                if s == -1:
                    trans = 0.0
                    pos = last_seen[i - 1][pi]
                else:
                    pos_prev = last_seen[i - 1][pi]
                    if pos_prev is None:
                        trans = 0.0
                    else:
                        d = np.hypot(cands[i][s][0] - pos_prev[0], cands[i][s][1] - pos_prev[1])
                        if d > BALL_MAX_PX_PER_FRAME * 6:
                            continue
                        trans = d * BALL_JUMP_COST
                    pos = cands[i][s][:2]
                if pc + trans < best:
                    best, bi = pc + trans, pi
            c_i[si] = best + unary if best < INF else INF
            b_i[si] = bi
            if best < INF:
                pos_i[si] = last_seen[i - 1][bi] if s == -1 else cands[i][s][:2]
        cost[i], back[i] = c_i, b_i
        last_seen[i] = pos_i
        prev_states = states

    # backtrack
    path = [None] * n
    si = int(np.argmin(cost[-1]))
    for i in range(n - 1, -1, -1):
        states = list(range(len(cands[i]))) + [-1]
        s = states[si]
        path[i] = None if s == -1 else cands[i][s]
        si = int(back[i][si])
    out = []
    for i, p in enumerate(path):
        if p is not None:
            out.append([i, p[0], p[1], p[2], False])
    # interpolate short gaps
    filled = []
    for a, b in zip(out, out[1:]):
        filled.append(a)
        gap = b[0] - a[0] - 1
        if 0 < gap <= BALL_MAX_GAP:
            for k in range(1, gap + 1):
                w = k / (gap + 1)
                filled.append([a[0] + k, a[1] + w * (b[1] - a[1]), a[2] + w * (b[2] - a[2]), 0.0, True])
    if out:
        filled.append(out[-1])
    return filled, int(len(static_cells))


PROBES = np.array([[480, 270, 1.0], [1440, 270, 1.0], [480, 810, 1.0], [1440, 810, 1.0], [960, 540, 1.0]])
HOMOGRAPHY_GLITCH_CM = 250
MEDIAN_WINDOW = 9


def repair_matrices(cache):
    """Return (matrices, n_replaced). Replaces matrices whose probe projections jump off the local median."""
    ms = [None if r["m"] is None else np.array(r["m"]) for r in cache]
    proj = np.full((len(ms), len(PROBES), 2), np.nan)
    for i, m in enumerate(ms):
        if m is not None:
            p = PROBES @ m.T
            proj[i] = p[:, :2] / p[:, 2:3]
    bad = np.zeros(len(ms), dtype=bool)
    h = MEDIAN_WINDOW // 2
    for i in range(len(ms)):
        if ms[i] is None:
            bad[i] = True
            continue
        win = proj[max(0, i - h): i + h + 1]
        med = np.nanmedian(win, axis=0)
        if np.max(np.linalg.norm(proj[i] - med, axis=1)) > HOMOGRAPHY_GLITCH_CM:
            bad[i] = True
    out, last = [], None
    first_good = next((m for m, b in zip(ms, bad) if not b), None)
    for m, b in zip(ms, bad):
        if not b:
            last = m
        out.append(last if last is not None else first_good)
    return out, int(bad.sum())


def project_xy(m, pts):
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if m is None or not len(pts):
        return np.zeros((0, 2))
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ m.T
    return p[:, :2] / p[:, 2:3]


def project(m, x, y):
    p = np.array(m) @ np.array([x, y, 1.0])
    return [float(p[0] / p[2]), float(p[1] / p[2])]


def main():
    rows = load("frames.jsonl")
    swaps, unresolved = repair_tracks(rows)
    print("swaps:", swaps)
    print("unresolved flips:", unresolved)
    before_ids = len({p[0] for r in rows for p in r["players"]})
    rows = apply_swaps(rows, swaps)
    team, gteam, votes = vote_teams(rows)

    per_frame = [{"frame": r["frame"],
                  "players": [[t, team[t], xy] for t, _, xy in r["players"]],
                  "goalkeepers": [[t, gteam.get(t, 0), xy] for t, _, xy in r["goalkeepers"]],
                  "referees": r["referees"]} for r in rows]
    cache_path = os.path.join(D, "cache.jsonl")
    cache = load("cache.jsonl") if os.path.exists(cache_path) else None
    n_glitch = 0
    if cache:
        mats, n_glitch = repair_matrices(cache)
        # re-project from pixel boxes with the repaired matrix; keep the voted teams
        for pf, c, m in zip(per_frame, cache, mats):
            by_id = {t[0]: t for t in c["tracks"]}
            for group, cls in (("players", 2), ("goalkeepers", 1)):
                for entry in pf[group]:
                    t = by_id.get(entry[0])
                    if t is not None and t[1] == cls:
                        xy = project_xy(m, [[(t[2] + t[4]) / 2, t[5]]])
                        if len(xy):
                            entry[2] = [round(float(xy[0][0]), 1), round(float(xy[0][1]), 1)]
    json.dump(per_frame, open(os.path.join(D, "tracks_clean.json"), "w"))

    summary = {"swaps": swaps, "unresolved_flips_overruled_by_vote": unresolved, "player_tracks_before": before_ids,
               "player_tracks_after": len(team),
               "tracks_with_any_minority_label": sum(1 for v in votes.values() if len(set(v)) > 1),
               "residual_noise_frames": int(sum(len(v) - Counter(v).most_common(1)[0][1] for v in votes.values())),
               "goalkeeper_tracks": {str(t): {"team": gteam.get(t), "frames": sum(1 for r in rows for g in r["goalkeepers"] if g[0] == t)}
                                     for t in sorted({g[0] for r in rows for g in r["goalkeepers"]})}}

    if cache:
        summary["homography_glitch_frames_repaired"] = n_glitch
        ball, n_static = track_ball(cache)
        out = []
        for i, x, y, conf, interp in ball:
            m = mats[i]
            if m is None:
                continue
            px, py = project(m, x, y)
            out.append({"frame": i, "px": [round(x, 1), round(y, 1)], "pitch_cm": [round(px, 1), round(py, 1)],
                        "conf": conf, "interpolated": interp})
        json.dump(out, open(os.path.join(D, "ball_clean.json"), "w"))
        real = [o for o in out if not o["interpolated"]]
        summary["ball"] = {"frames_total": len(cache), "frames_with_ball_real": len(real),
                           "frames_with_ball_incl_interpolated": len(out), "static_cells_removed": n_static,
                           "frames_with_any_candidate": sum(1 for r in cache if r["ball"])}
    print(json.dumps(summary, indent=2))
    json.dump(summary, open(os.path.join(D, "postprocess_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
