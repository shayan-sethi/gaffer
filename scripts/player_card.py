#!/usr/bin/env python3
"""
Per-player metric card from a clip_ids.py run (uses its cached detections; no re-detection).

  python player_card.py <video> <out_dir> <start_s> <seconds> <n_players> [identity=random] [seed]

random = a random pick among the identities whose shirt colour stays consistent (an identity that mixes
light and dark shirts has probably merged two people, so its numbers would be meaningless).
Writes <out_dir>/card_<k>.json and card_<k>.png.

Distances/speeds are ESTIMATES: metres per pixel from the player's own box height (1.75 m) and an assumed
30 degree camera elevation; no pitch calibration. "left/middle/right" is the camera view, not the pitch.
"""
import sys, json, random, math
from pathlib import Path
import numpy as np
import cv2

video, out, start_s, seconds, n = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
pick = sys.argv[6] if len(sys.argv) > 6 else "random"
seed = int(sys.argv[7]) if len(sys.argv) > 7 else None
sys.argv = ["clip_ids.py", video, str(out), start_s, seconds, n]
sys.path.insert(0, str(Path(__file__).parent))
import clip_ids as C                                   # noqa: E402  (module-level argv parsing only)

H_M, ELEV = 1.75, math.radians(30)
MAX_STEP_MS = 9.0                                      # faster than this between samples = box/ID jump
ZONES = [("standing", 0, 0.5), ("walking", 0.5, 1.94), ("jogging", 1.94, 3.89), ("running", 3.89, 5.83), ("sprinting", 5.83, 99)]


def sm(a, w=5):
    p = np.pad(a, (w // 2, w // 2), mode="edge")
    return np.array([np.median(p[i:i + w]) for i in range(len(a))])


def runs_at_least(mask, n_min):
    runs, cur = 0, 0
    for v in list(mask) + [False]:
        if v:
            cur += 1
        else:
            runs += cur >= n_min
            cur = 0
    return int(runs)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = json.load(open(out / "meta.json"))
    rows = sorted((json.loads(l) for l in open(out / "dets.jsonl")), key=lambda r: r["frame"])
    sfps = meta["fps"] / meta["stride"]
    W, Hh = meta["width"], meta["height"]
    per_frame, _ = C.track(rows, np.ones((Hh, W), np.uint8), [], sfps)
    ident, unassigned, *_ = C.lock_identities(per_frame, sfps)
    ident.sort(key=lambda I: -len(I["frames"]))
    clean = []
    for k, I in enumerate(ident):
        b = np.array([np.mean(o["rgb"]) for o in I["frames"].values() if not o.get("interp")])
        if np.percentile(b, 90) - np.percentile(b, 10) < 80:
            clean.append(k)
    print("identities with a consistent shirt colour:", [k + 1 for k in clean])
    k = random.Random(seed).choice(clean) if pick == "random" else int(pick) - 1
    I = ident[k]
    fr = np.array(sorted(I["frames"]))
    t = fr / sfps
    feet = np.array([C.foot(I["frames"][f]) for f in fr])
    hpx = sm(np.array([C.height(I["frames"][f]) for f in fr]), 9)
    fx, fy = sm(feet[:, 0]), sm(feet[:, 1])
    sx, sy = H_M * math.cos(ELEV) / hpx, H_M / (hpx * math.tan(ELEV))
    dt = np.diff(t)
    step_m = np.hypot(np.diff(fx) * sx[1:], np.diff(fy) * sy[1:])
    step_v = step_m / np.maximum(dt, 1e-6)
    valid = (dt <= 2.0 / sfps + 1e-6) & (step_v <= MAX_STEP_MS)
    dist = float(step_m[valid].sum())
    inst = np.where(valid, step_v, np.nan)
    # 1-second rolling mean speed (needs >= 60% valid steps in the window)
    w = int(round(sfps))
    sp = np.full(len(fr), np.nan)
    for i in range(len(inst)):
        win = inst[max(0, i - w + 1):i + 1]
        if np.isfinite(win).sum() >= 0.6 * w:
            sp[i + 1] = float(np.nanmean(win))
    ok = np.isfinite(sp)
    zone_pct = {name: round(100 * float(np.mean((sp[ok] >= lo) & (sp[ok] < hi)))) for name, lo, hi in ZONES}
    top = float(np.nanmax(sp)) if ok.any() else float("nan")
    sprints = runs_at_least(np.nan_to_num(sp, nan=0) >= 5.83, int(0.8 * sfps))
    xs = fx / W
    secs = int(math.ceil(t.max() / 10))
    per10 = [float(step_m[valid & (t[1:] >= 10 * s) & (t[1:] < 10 * (s + 1))].sum()) for s in range(secs)]
    card = dict(
        identity=k + 1, team=I["team"], seen_seconds=round(len(fr) / sfps, 1), seen_pct=round(100 * len(fr) / len(rows)),
        distance_m=round(dist, 1), distance_per_min_m=round(dist / (len(fr) / sfps) * 60),
        avg_speed_kmh=round(float(np.nanmean(sp)) * 3.6, 1), top_speed_kmh=round(top * 3.6, 1),
        zone_time_pct=zone_pct, sprints=sprints,
        camera_view_thirds_pct=dict(left=round(100 * float(np.mean(xs < 1 / 3))),
                                    middle=round(100 * float(np.mean((xs >= 1 / 3) & (xs < 2 / 3)))),
                                    right=round(100 * float(np.mean(xs >= 2 / 3)))),
        distance_per_10s_m=[round(v, 1) for v in per10],
        jumps_removed=int((~valid & (step_v > MAX_STEP_MS)).sum()),
        note="distances/speeds are rough estimates (no pitch calibration)")
    (out / f"card_{k + 1}.json").write_text(json.dumps(card, indent=1))
    print(json.dumps(card, indent=1))

    # ---- picture
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, meta["start_frame"] + int(len(rows) * 0.5 * meta["stride"]))
    _, bg = cap.read()
    heat = np.zeros((Hh, W), np.float32)
    for x, y in zip(fx, fy):
        if 0 <= int(y) < Hh and 0 <= int(x) < W:
            heat[int(y), int(x)] += 1
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(12, W / 40))
    heat = heat / max(heat.max(), 1e-6)
    col = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    a = (np.clip(heat * 1.6, 0, 0.75))[..., None]
    comp = (bg * (1 - a) + col * a).astype(np.uint8)
    for i in range(1, len(fr)):
        if valid[i - 1]:
            cv2.line(comp, (int(fx[i - 1]), int(fy[i - 1])), (int(fx[i]), int(fy[i])), (255, 255, 255), 1)
    cv2.circle(comp, (int(fx[0]), int(fy[0])), 6, (255, 255, 255), -1)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 9.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[3.2, 1.3, 1.3], hspace=0.45, wspace=0.28)
    ax = fig.add_subplot(gs[0, :2])
    ax.imshow(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Where player #{k + 1} (team {I['team']}) spent the minute  -  heat on the camera view, white line = path", fontsize=11)
    ax.axis("off")
    tx = fig.add_subplot(gs[0, 2])
    tx.axis("off")
    lines = [("On screen", f"{card['seen_seconds']} s  ({card['seen_pct']}%)"), ("Distance (est.)", f"{card['distance_m']} m"),
             ("Per minute", f"{card['distance_per_min_m']} m/min"), ("Average speed", f"{card['avg_speed_kmh']} km/h"),
             ("Top speed (1 s)", f"{card['top_speed_kmh']} km/h"), ("Sprints (>21 km/h, 0.8 s+)", f"{sprints}"),
             ("Camera view L / M / R", f"{card['camera_view_thirds_pct']['left']} / {card['camera_view_thirds_pct']['middle']} / {card['camera_view_thirds_pct']['right']} %")]
    for i, (a_, b_) in enumerate(lines):
        tx.text(0, 1 - i * 0.12, a_, fontsize=11, color="#9aa", va="top")
        tx.text(0, 1 - i * 0.12 - 0.05, b_, fontsize=15, color="white", va="top", fontweight="bold")
    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(t, np.where(ok, sp * 3.6, np.nan), color="#ffd54a", lw=1.6)
    for name, lo, hi in ZONES[1:]:
        ax1.axhline(lo * 3.6, color="#556", lw=0.7, ls="--")
    ax1.set_ylabel("km/h"); ax1.set_xlabel("seconds into the clip"); ax1.set_title("Speed (1-second average, estimated)", fontsize=10)
    ax2 = fig.add_subplot(gs[2, :2])
    ax2.bar(np.arange(secs) * 10 + 5, per10, width=8, color="#4fc3f7")
    ax2.set_ylabel("metres"); ax2.set_xlabel("seconds into the clip"); ax2.set_title("Distance covered per 10 seconds (estimated)", fontsize=10)
    ax3 = fig.add_subplot(gs[2, 2])
    names = [z[0] for z in ZONES]
    ax3.barh(names, [zone_pct[nm] for nm in names], color="#81c784")
    ax3.invert_yaxis(); ax3.set_xlabel("% of time"); ax3.set_title("Time by speed zone", fontsize=10)
    fig.text(0.5, 0.005, "Rough estimates: no pitch calibration (assumes 1.75 m player height and a 30 degree camera angle). Use for comparing players, not as exact numbers.",
             ha="center", fontsize=8, color="#889")
    fig.savefig(out / f"card_{k + 1}.png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", out / f"card_{k + 1}.png")


if __name__ == "__main__":
    main()
