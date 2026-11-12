#!/usr/bin/env python3
"""
Ball tracker for a fixed camera: Kalman filter + gating + confirmation + RTS smoothing, and the simulated-occlusion test.
Follows the output contract of the build plan: one row per ball step with frame, t, x, y, conf, source (det|pred|interp|lost), gap_len.

  python ball_kalman.py track <out_dir> [sfps=10]            reads <out_dir>/ball.jsonl (+ dets.jsonl for row numbering)  -> ball_track.csv
  python ball_kalman.py eval  <out_dir> [sfps=10]            simulated occlusion test on reliably-detected stretches       -> gap_error.json

All thresholds are in CFG (override with env BK_<NAME>, e.g. BK_GATE0=90). Units: ball steps (one step = one row of ball.jsonl = 0.2 s at sfps 10) and px.
"""
import sys, os, json, csv, math
from pathlib import Path
import numpy as np

CFG = dict(
    DT=1.0,               # one step
    Q_ACC=22.0,           # process noise: std of the ball's acceleration change per step (px/step^2) - a kick changes velocity a lot
    R_PX=4.0,             # measurement noise (px)
    GATE0=70.0,           # gate radius at 0 missed steps (px)
    GATE_GROW=45.0,       # extra radius per missed step (px)
    GATE_MAX=330.0,
    MIN_CONF=0.15,        # candidates below this are ignored entirely
    CONFIRM_CONF=0.35,    # a candidate that can START a track must be at least this confident
    CONFIRM_DIST=90.0,    # ... and be followed by a consistent one in the next 1-2 steps within this distance (px per step)
    MAX_MISS=5,           # steps (1 s) the tracker keeps predicting before it declares the ball LOST
    INTERP_MAX_GAP=3,     # gaps up to this many steps are filled from the smoothed track and labelled 'interp'; longer ones 'pred'
    SCORE_DIST_W=0.5,     # candidate score = 100*conf - SCORE_DIST_W * distance_to_prediction
    COAST_DECAY=0.80,     # confidence multiplier per predicted step
)
for k in list(CFG):
    v = os.environ.get("BK_" + k)
    if v is not None:
        CFG[k] = float(v)

F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)


def Qmat(q):
    a = np.array([[0.25, 0.5], [0.5, 1.0]]) * q * q
    Q = np.zeros((4, 4)); Q[np.ix_([0, 2], [0, 2])] = a; Q[np.ix_([1, 3], [1, 3])] = a
    return Q


HI = os.environ.get("BK_HI") == "1"           # high-rate ball file (ball_hi.jsonl): one step = a few video frames, parameters are rescaled from the 0.2 s defaults
if HI:
    _S = float(os.environ.get("BK_STEP_S", 0.04)) / 0.2                      # step length relative to 0.2 s
    CFG.update(Q_ACC=max(3.0, 22.0 * _S ** 0.5 * _S ** 0.5 * 4), GATE0=max(18.0, 70.0 * _S * 1.4), GATE_GROW=45.0 * _S + 4.0, CONFIRM_DIST=90.0 * _S + 6.0,
               MAX_MISS=int(round(5 / _S)), INTERP_MAX_GAP=int(round(3 / _S)), COAST_DECAY=0.80 ** _S)
    for k in list(CFG):
        v = os.environ.get("BK_" + k)
        if v is not None:
            CFG[k] = float(v)
QM = Qmat(CFG["Q_ACC"]); RM = np.eye(2) * CFG["R_PX"] ** 2


def load(D, sfps):
    D = Path(D)
    rows = sorted(json.loads(l)["frame"] for l in open(D / "dets.jsonl")); idx = {f: i for i, f in enumerate(rows)}
    if HI:
        meta = json.load(open(D / "meta.json")); per_row = meta["stride"]; first = rows[0]
        idx = {b["frame"]: (b["frame"] - first) / per_row for b in map(json.loads, open(D / "ball_hi.jsonl"))}     # fractional row index = time * sfps
    B = [json.loads(l) for l in open(D / ("ball_hi.jsonl" if HI else "ball.jsonl"))]
    fx = json.load(open(D / "ball_fixtures.json")) if (D / "ball_fixtures.json").exists() else []       # spots that fire all match long (see ball_fixtures.py): never the ball
    if fx:
        n0 = sum(len(b["cands"]) for b in B)
        for b in B:
            b["cands"] = [c for c in b["cands"] if all(abs((c[0] + c[2]) / 2 - f["x"]) > f["r"] or abs((c[1] + c[3]) / 2 - f["y"]) > f["r"] for f in fx)]
        print(f"fixtures: dropped {n0 - sum(len(b['cands']) for b in B)} candidates at {len(fx)} fixed spots", flush=True)
    if (D / "identities.json").exists() and os.environ.get("BK_DORMANT", "1") == "1":
        ids_ = json.load(open(D / "identities.json")); fps_b = 1.0 / max(np.median(np.diff(sorted(idx[b["frame"]] for b in B))) / sfps, 1e-6) if len(B) > 2 else 5.0
        Wn = max(1, int(round(0.5 * fps_b))); BF = sorted(B, key=lambda b: b["frame"]); dropped = 0
        cs = [[((c[0] + c[2]) / 2, (c[1] + c[3]) / 2) for c in b["cands"]] for b in BF]
        for i, b in enumerate(BF):
            row = int(round(idx[b["frame"]])); players = []
            for I in ids_.values():
                bx = I["rows"].get(str(row)) or I["rows"].get(str(row - 1)) or I["rows"].get(str(row + 1))
                if bx:
                    players.append(((bx[0] + bx[2]) / 2, bx[3] - 0.1 * (bx[3] - bx[1]), bx[3] - bx[1]))
            keep = []
            for k, c in enumerate(b["cands"]):
                x, y = cs[i][k]
                win = range(max(0, i - Wn), min(len(BF), i + Wn + 1))
                static = sum(1 for j in win if any(abs(x - px) <= 8 and abs(y - py) <= 8 for px, py in cs[j])) >= 0.6 * len(win)
                lonely = players and all(np.hypot(x - px, y - py) / ph > 1.5 for px, py, ph in players)
                if static and lonely:
                    dropped += 1
                else:
                    keep.append(c)
            b["cands"] = keep
        print(f"dormant filter: dropped {dropped} stationary candidates with no player within 1.5 body-heights", flush=True)
    steps = []                                  # (row_index, [(x, y, conf), ...])
    for b in sorted(B, key=lambda b: b["frame"]):
        steps.append((idx[b["frame"]], [((c[0] + c[2]) / 2, (c[1] + c[3]) / 2, c[4]) for c in b["cands"] if c[4] >= CFG["MIN_CONF"]]))
    return steps


def start_track(steps, k):
    """a candidate at step k that is confident AND followed by a consistent one at k+1 or k+2 -> initial state, else None"""
    for (x, y, c) in steps[k][1]:
        if c < CFG["CONFIRM_CONF"]:
            continue
        for dk in (1, 2):
            if k + dk >= len(steps):
                break
            for (x2, y2, c2) in steps[k + dk][1]:
                if c2 >= CFG["CONFIRM_CONF"] * 0.8 and math.hypot(x2 - x, y2 - y) <= CFG["CONFIRM_DIST"] * dk:
                    v = np.array([(x2 - x) / dk, (y2 - y) / dk])
                    return np.array([x, y, v[0], v[1]]), c
    return None


def forward(steps):
    """Kalman forward pass. Returns segments: each a list of dicts {k, xp, Pp, xf, Pf, det(None|(x,y,c)), miss}"""
    n = len(steps); segs = []; k = 0
    while k < n:
        st = start_track(steps, k)
        if st is None:
            k += 1; continue
        x, c0 = st; P = np.diag([CFG["R_PX"] ** 2, CFG["R_PX"] ** 2, 40.0 ** 2, 40.0 ** 2]); seg = []; miss = 0
        seg.append(dict(k=k, xp=x.copy(), Pp=P.copy(), xf=x.copy(), Pf=P.copy(), det=steps[k][1][0] if steps[k][1] else None, miss=0))
        kk = k + 1
        while kk < n:
            xp = F @ x; Pp = F @ P @ F.T + QM
            gate = min(CFG["GATE_MAX"], CFG["GATE0"] + CFG["GATE_GROW"] * miss)
            best = None
            for (cx, cy, cc) in steps[kk][1]:
                d = math.hypot(cx - xp[0], cy - xp[1])
                if d <= gate:
                    sc = 100.0 * cc - CFG["SCORE_DIST_W"] * d
                    if best is None or sc > best[0]:
                        best = (sc, cx, cy, cc)
            if best is not None:
                z = np.array([best[1], best[2]]); S = H @ Pp @ H.T + RM; K = Pp @ H.T @ np.linalg.inv(S)
                x = xp + K @ (z - H @ xp); P = (np.eye(4) - K @ H) @ Pp; miss = 0
                seg.append(dict(k=kk, xp=xp, Pp=Pp, xf=x.copy(), Pf=P.copy(), det=(best[1], best[2], best[3]), miss=0))
            else:
                x, P = xp, Pp; miss += 1
                seg.append(dict(k=kk, xp=xp, Pp=Pp, xf=x.copy(), Pf=P.copy(), det=None, miss=miss))
                if miss > CFG["MAX_MISS"]:
                    break
            kk += 1
        while seg and seg[-1]["det"] is None:            # trailing predictions with no later detection: the ball was lost there
            seg.pop()
        if len(seg) >= 2:
            segs.append(seg)
        k = (seg[-1]["k"] + 1) if seg else k + 1
        k = max(k, kk - CFG["MAX_MISS"] if kk - CFG["MAX_MISS"] > k else k)
    return segs


def rts(seg):
    """backward RTS smoothing within a segment; returns smoothed states"""
    n = len(seg); xs = [s["xf"].copy() for s in seg]; Ps = [s["Pf"].copy() for s in seg]
    for i in range(n - 2, -1, -1):
        Pf = seg[i]["Pf"]; Pn = seg[i + 1]["Pp"]
        try:
            G = Pf @ F.T @ np.linalg.inv(Pn)
        except np.linalg.LinAlgError:
            continue
        xs[i] = seg[i]["xf"] + G @ (xs[i + 1] - seg[i + 1]["xp"]); Ps[i] = Pf + G @ (Ps[i + 1] - Pn) @ G.T
    return xs, Ps


def track(steps):
    """returns per-step records: k -> dict(x, y, conf, source, gap_len)"""
    out = {}
    for seg in forward(steps):
        xs, _ = rts(seg); last_det = None; conf = 1.0
        for s, xsm in zip(seg, xs):
            k = s["k"]
            if s["det"] is not None:
                out[k] = dict(x=s["det"][0], y=s["det"][1], conf=s["det"][2], source="det", gap_len=0); last_det = k; conf = s["det"][2]
            else:
                gl = k - last_det if last_det is not None else 0
                nxt = next((q["k"] for q in seg if q["k"] > k and q["det"] is not None), None)
                total_gap = (nxt - last_det - 1) if (nxt is not None and last_det is not None) else gl
                out[k] = dict(x=float(xsm[0]), y=float(xsm[1]), conf=conf * CFG["COAST_DECAY"] ** gl,
                              source="interp" if total_gap <= CFG["INTERP_MAX_GAP"] else "pred", gap_len=gl)
    return out


def write_csv(D, steps, rec, sfps):
    path = Path(D) / "ball_track.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "t", "x", "y", "conf", "source", "gap_len"])
        last = None
        for k, (row, _) in enumerate(steps):
            r = rec.get(k)
            if r is None:
                last_gap = 0
                w.writerow([round(row, 2), round(row / sfps, 3), "", "", 0, "lost", ""])
            else:
                w.writerow([round(row, 2), round(row / sfps, 3), round(r["x"], 1), round(r["y"], 1), round(r["conf"], 3), r["source"], r["gap_len"]])
    return path


def cmd_track(D, sfps):
    steps = load(D, sfps); rec = track(steps); p = write_csv(D, steps, rec, sfps)
    from collections import Counter
    c = Counter(r["source"] for r in rec.values()); n = len(steps)
    print(f"{n} ball steps: det {c['det']}, interp {c['interp']}, pred {c['pred']}, lost {n - len(rec)}  (tracked {100 * len(rec) / n:.0f}%)  -> {p}")
    segs = []; s = None
    for k in range(n):
        if k in rec:
            s = k if s is None else s
        elif s is not None:
            segs.append((s, k - 1)); s = None
    if s is not None: segs.append((s, n - 1))
    print("tracked stretches (s):", [(round(steps[a][0] / sfps, 1), round(steps[b][0] / sfps, 1)) for a, b in segs])


def cmd_eval(D, sfps):
    """Simulated occlusion: take stretches where the ball is reliably detected on consecutive steps (conf >= 0.5 and one
    consistent candidate), hide K consecutive detections in the middle, and measure how far the prediction is from the hidden
    detections. Compared: (a) forward Kalman coasting only, (b) Kalman + RTS (uses the detection after the gap), (c) straight line
    between the last and next detection (what my old gap filling did)."""
    steps = load(D, sfps); n = len(steps)
    good = {}
    for k, (row, cs) in enumerate(steps):
        c = [c for c in cs if c[2] >= 0.5]
        if len(c) == 1: good[k] = c[0]
    ball_d = 14.0; res = {}
    for K in (1, 2, 3, 5, 8):
        errs = {"kalman_forward": [], "kalman_rts": [], "straight_line": []}
        for k0 in range(2, n - K - 3):
            ks = list(range(k0 - 2, k0 + K + 3))
            if not all(q in good for q in ks): continue
            ab = [good[q] for q in ks]
            if max(math.hypot(ab[i + 1][0] - ab[i][0], ab[i + 1][1] - ab[i][1]) for i in range(len(ab) - 1)) < 6.0: continue     # skip a ball at rest
            hid = list(range(k0, k0 + K))
            masked = [(steps[q][0], ([] if q in hid else [good[q]])) for q in ks]
            # forward-only: filter on the 2 steps before, then coast
            x = np.array([good[k0 - 2][0], good[k0 - 2][1], good[k0 - 1][0] - good[k0 - 2][0], good[k0 - 1][1] - good[k0 - 2][1]], float)
            P = np.diag([CFG["R_PX"] ** 2] * 2 + [30.0 ** 2] * 2); xs_f = []; xx, PP = x, P
            xx = np.array([good[k0 - 1][0], good[k0 - 1][1], x[2], x[3]], float)
            for q in hid:
                xx = F @ xx; xs_f.append(xx[:2].copy())
            # RTS: run the full forward on the masked stretch (start_track needs confirmation) and smooth
            seg_steps = [(r, c) for r, c in masked]; recs = track(seg_steps)
            smooth = [np.array([recs[q - (k0 - 2)]["x"], recs[q - (k0 - 2)]["y"]]) if (q - (k0 - 2)) in recs else None for q in hid]
            a_, b_ = good[k0 - 1], good[k0 + K]
            for j, q in enumerate(hid):
                truth = np.array(good[q][:2]); t = (j + 1) / (K + 1)
                errs["kalman_forward"].append(np.linalg.norm(xs_f[j] - truth))
                if smooth[j] is not None: errs["kalman_rts"].append(np.linalg.norm(smooth[j] - truth))
                errs["straight_line"].append(np.linalg.norm(np.array(a_[:2]) * (1 - t) + np.array(b_[:2]) * t - truth))
        res[K] = {m: dict(n=len(v), mean_px=round(float(np.mean(v)), 1) if v else None, median_px=round(float(np.median(v)), 1) if v else None,
                         within_1_5_balls=round(float(np.mean(np.array(v) < 1.5 * ball_d)), 2) if v else None) for m, v in errs.items()}
    json.dump(res, open(Path(D) / "gap_error.json", "w"), indent=1)
    print("gap length (steps of 0.2 s) -> mean error in px (share of predictions within 1.5 ball widths = 21 px)")
    for K, r in res.items():
        print(f"  K={K} ({K * 0.2:.1f} s hidden):", "  ".join(f"{m}: {v['mean_px']} px ({v['within_1_5_balls']})  n={v['n']}" for m, v in r.items()))


if __name__ == "__main__":
    cmd, D = sys.argv[1], sys.argv[2]; sf = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    {"track": cmd_track, "eval": cmd_eval}[cmd](D, sf)
