#!/usr/bin/env python3
"""
Touch-based passes. Reads ball_track.csv (ball_kalman.py) + identities.json and writes pass_events.json.

A player HAS the ball at a tracked step when he is the nearest player (feet to ball, in body-heights) within HOLD_D AND clearly nearer than
the next player (margin >= MARGIN). Consecutive such steps by one player form a SPELL (up to 1 unclear step allowed inside). A handover is the
change from one spell to the next spell of a DIFFERENT player:
  same team -> pass, other team -> turnover.
How sure we are:
  seen      ball tracked the whole way between the two spells (gap <= SEEN_GAP_S) and it moved at least MIN_TRAVEL body-heights
  probable  ball hidden between them for <= PROB_GAP_S
  (longer gaps are not reported as events)
Nothing here needs the ball to be held still, and nothing depends on a minimum holding time: a one-touch return counts.

  python pass_events.py <out_dir> [sfps=10]     env: HOLD_D=0.75 MARGIN=0.2 SEEN_GAP_S=0.6 PROB_GAP_S=1.8 MIN_TRAVEL=0.5
"""
import sys, os, json, csv
from pathlib import Path
import math
import numpy as np

HOLD_D = float(os.environ.get("HOLD_D", 0.75)); MARGIN = float(os.environ.get("MARGIN", 0.2))
SEEN_GAP_S = float(os.environ.get("SEEN_GAP_S", 0.6)); PROB_GAP_S = float(os.environ.get("PROB_GAP_S", 1.8)); MIN_TRAVEL = float(os.environ.get("MIN_TRAVEL", 0.5))
LONG_GAP_S = float(os.environ.get("LONG_GAP_S", 4.0))
CHALLENGE_MAX_S = float(os.environ.get("CHALLENGE_MAX_S", 1.2)); RESUME_S = float(os.environ.get("RESUME_S", 1.5))
TOUCH_REL = float(os.environ.get("TOUCH_REL", 0.35))      # velocity change needed for a touch, as a share of the ball's speed
RT_GAP_S = float(os.environ.get("RT_GAP_S", 1.0)); RT_NEAR = float(os.environ.get("RT_NEAR", 1.0))      # round trip: hidden at least this long, teammate within this many body-heights of where the ball reappears
DROP_TRAVEL = float(os.environ.get("DROP_TRAVEL", 0.1))      # handover with the ball moving less than this many body-heights = two players at one ball, not a pass
TOUCH_SLOW = float(os.environ.get("TOUCH_SLOW", 22)); TOUCH_DV = float(os.environ.get("TOUCH_DV", 28))      # px per step (0.2 s)


def run(D, sfps=10.0, start_s=0.0):
    """Works in TICKS = steps of the ball track (5 per second on the old 0.2 s track, 25 per second on a high-rate track). Player boxes are
    interpolated between their own (10 per second) rows. Thresholds are in seconds / body-heights / px per 0.2 s, so both rates behave alike."""
    D = Path(D)
    ids = json.load(open(D / "identities.json"))
    raw = [(float(r["t"]), float(r["x"]), float(r["y"])) for r in csv.DictReader(open(D / "ball_track.csv")) if r["x"] != "" and r["source"] in ("det", "interp")]
    allt = [float(r["t"]) for r in csv.DictReader(open(D / "ball_track.csv"))]
    if len(raw) < 2:
        return []
    dts = np.diff(sorted(set(allt))); TPS = 1.0 / float(np.median(dts)) if len(dts) else 5.0
    LAG = max(1, int(round(0.2 * TPS)))                                  # ticks in 0.2 s
    trk = {int(round(t * TPS)): (x, y) for t, x, y in raw}
    ks = sorted(trk)
    prow = {I["label"]: {int(k): v for k, v in I["rows"].items()} for I in ids.values()}

    def box(lbl, tick):
        r = tick / TPS * sfps; r0 = int(math.floor(r)); d = prow[lbl]; f = r - r0
        if r0 in d and r0 + 1 in d:
            return [d[r0][j] * (1 - f) + d[r0 + 1][j] * f for j in range(4)]
        for q in (r0, r0 + 1):
            if q in d:
                return d[q][:4]
        return None

    def rank(tick, p):
        out = []
        for I in ids.values():
            if I["role"] == "REF":
                continue
            b = box(I["label"], tick)
            if b:
                h = b[3] - b[1]
                out.append((float(np.hypot(p[0] - (b[0] + b[2]) / 2, p[1] - (b[3] - 0.1 * h)) / h), I["label"], I["team"], h))
        return sorted(out)

    def near(k, d):                                                       # nearest tracked tick to k+d within 1 tick
        for e in (0, 1, -1):
            if k + d + e in trk:
                return k + d + e
        return None

    def vels(k):
        prv, nxt = near(k, -LAG), near(k, LAG)
        v_in = (np.array(trk[k]) - np.array(trk[prv])) / max(k - prv, 1) * LAG if prv is not None else None
        v_out = (np.array(trk[nxt]) - np.array(trk[k])) / max(nxt - k, 1) * LAG if nxt is not None else None
        return v_in, v_out

    dvmap = {}
    for k in ks:
        a_, b_ = vels(k)
        dvmap[k] = float(np.linalg.norm(b_ - a_)) if (a_ is not None and b_ is not None) else None

    def touched(k):
        """a TOUCH changes the ball's velocity (speed or direction) or leaves it nearly stopped. Velocities are px per 0.2 s measured over +-0.2 s.
        The velocity change must be the PEAK within +-0.2 s: the ticks next to a kick see the same acceleration but belong to the kicker, not to whoever is nearest."""
        v_in, v_out = vels(k)
        sp = [np.linalg.norm(v) for v in (v_in, v_out) if v is not None]
        if sp and min(sp) <= TOUCH_SLOW:
            return True
        d = dvmap.get(k)
        if d is not None and d >= max(TOUCH_DV, TOUCH_REL * max(np.linalg.norm(v_in), np.linalg.norm(v_out))):
            if d >= max((dvmap.get(j) or 0.0) for j in range(k - LAG, k + LAG + 1)):
                return True
        return False

    holder = {}
    for k in ks:
        r = rank(k, trk[k])
        if r and r[0][0] <= HOLD_D and (len(r) == 1 or r[1][0] - r[0][0] >= MARGIN) and touched(k):
            holder[k] = r[0][1]
    team = {I["label"]: I["team"] for I in ids.values()}
    hgt = {I["label"]: float(np.median([v[3] - v[1] for v in I["rows"].values()])) for I in ids.values()}
    spells = []                                                           # [label, first_tick, last_tick]
    for k in ks:
        h = holder.get(k)
        if h is None:
            continue
        if spells and spells[-1][0] == h and k - spells[-1][2] <= LAG:
            spells[-1][2] = k
        else:
            spells.append([h, k, k])
    contested = []
    changed = True
    while changed:                                                        # a short opponent spell between two spells of one team = a challenge, not a loss
        changed = False
        for i in range(1, len(spells) - 1):
            p, b, n = spells[i - 1], spells[i], spells[i + 1]
            if (team[b[0]] != team[p[0]] and team[n[0]] == team[p[0]] and (b[2] - b[1]) / TPS <= CHALLENGE_MAX_S and (n[1] - b[2]) / TPS <= RESUME_S):
                contested.append(dict(t=round(start_s + b[1] / TPS, 2), opponent=b[0], between=[p[0], n[0]]))
                del spells[i]
                if spells[i - 1][0] == spells[i][0]:
                    spells[i - 1][2] = spells[i][2]; del spells[i]
                changed = True
                break
    rowof = lambda tk: int(round(tk / TPS * sfps))
    json.dump([dict(label=a[0], team=team[a[0]], first=rowof(a[1]), last=rowof(a[2])) for a in spells], open(D / "pass_spells.json", "w"))
    json.dump(contested, open(D / "pass_contested.json", "w"), indent=1)
    events = []
    for a, b in zip(spells, spells[1:]):
        if a[0] == b[0]:
            # ROUND TRIP: the same player has the ball before and after a long hidden gap. A ball that leaves a player and comes back to him was played by someone else
            # on the way - if a teammate is close to the ball where it reappears, that teammate played it: two INFERRED passes (out and back).
            hid = sum(1 for k in range(a[2] + 1, b[1]) if k not in trk) / TPS
            if hid >= RT_GAP_S and b[1] in trk:
                cand = [r for r in rank(b[1], trk[b[1]]) if r[1] != a[0] and r[2] == team[a[0]] and r[0] <= RT_NEAR]
                if cand:
                    p = cand[0][1]
                    events.append(dict(row=rowof(a[2]), t=round(start_s + a[2] / TPS, 2), kind="pass", team=team[a[0]], frm=a[0], to=p, gap_s=round(hid, 1), ball_travel_px=0, frm_row=rowof(a[2]), confidence="inferred"))
                    events.append(dict(row=rowof(b[1]), t=round(start_s + b[1] / TPS, 2), kind="pass", team=team[a[0]], frm=p, to=b[0], gap_s=0.0, ball_travel_px=0, frm_row=rowof(b[1]), confidence="inferred"))
            continue
        gap_s = sum(1 for k in range(a[2] + 1, b[1]) if k not in trk) / TPS
        pa, pb = trk[a[2]], trk[b[1]]
        travel = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])); h = hgt[a[0]]
        same_team = team[a[0]] == team[b[0]]
        if gap_s > (LONG_GAP_S if same_team else PROB_GAP_S):
            continue
        conf = "seen" if (gap_s <= SEEN_GAP_S and travel >= MIN_TRAVEL * h) else "probable"
        if (travel < DROP_TRAVEL * h and (b[1] - a[2]) <= 1) or ((b[1] - a[2]) / TPS < 0.2 and travel < 1.0 * h):
            continue                          # touches by different players <0.2 s apart without the ball travelling = one contact, not a pass
        events.append(dict(row=rowof(b[1]), t=round(start_s + b[1] / TPS, 2), kind="pass" if same_team else "turnover", team=team[a[0]], frm=a[0], to=b[0],
                           gap_s=round(gap_s, 1), ball_travel_px=round(travel), frm_row=rowof(a[2]), confidence=conf))
    events.sort(key=lambda e: e["t"])
    return events


if __name__ == "__main__":
    D = sys.argv[1]; sf = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0; st = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    ev = run(D, sf, st)
    json.dump(ev, open(Path(D) / "pass_events.json", "w"), indent=1)
    n = {k: sum(1 for e in ev if e["kind"] == k) for k in ("pass", "turnover")}
    print(f"{len(ev)} events: {n['pass']} passes, {n['turnover']} turnovers | seen {sum(e['confidence'] == 'seen' for e in ev)}, probable {sum(e['confidence'] == 'probable' for e in ev)}")
    for e in ev:
        print(f"  {e['t'] - st:5.1f}s  {e['kind']:9s} {e['frm']:>3s} -> {e['to']:<3s}  {e['confidence']:8s} hidden {e['gap_s']} s, ball moved {e['ball_travel_px']} px")
    br = [(a["t"] - st, a["to"], b["frm"]) for a, b in zip(ev, ev[1:]) if a["to"] != b["frm"]]
    if br:
        print("possession chain breaks (a receiver who never passes on, or a passer who never received - something in between was missed):")
        for t, a, b in br:
            print(f"  after {t:.1f}s: ball went to {a}, next event starts from {b}")
