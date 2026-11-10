#!/usr/bin/env python3
"""One page (PNG) per player from player_series.pkl.   python player_pages.py <out_dir> [pitch_len=105] [pitch_wid=68]"""
import sys, pickle, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
OUT = Path(sys.argv[1]); PL = float(sys.argv[2]) if len(sys.argv) > 2 else 105.0; PW = float(sys.argv[3]) if len(sys.argv) > 3 else 68.0
res = pickle.load(open(OUT / "player_series.pkl", "rb"))
(OUT / "players").mkdir(exist_ok=True)
plt.style.use("dark_background")
ZONES = ["standing", "walking", "jogging", "running", "sprinting"]
for r in res:
    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.2], width_ratios=[2.2, 1, 1], hspace=0.35, wspace=0.3)
    ax = fig.add_subplot(gs[0, :2])
    ax.add_patch(Rectangle((0, 0), PL, PW, color="#2e6b3a", zorder=0))
    # heat: where the player spent time
    H, xe, ye = np.histogram2d(r["_X"], r["_Y"], bins=[int(PL / 3), int(PW / 3)], range=[[0, PL], [0, PW]])
    from scipy.ndimage import gaussian_filter
    H = gaussian_filter(H, 1.2)
    if H.max() > 0:
        ax.imshow(np.ma.masked_less(H.T, 0.04 * H.max()), origin="lower", extent=[0, PL, 0, PW], cmap="YlOrRd", alpha=0.6, zorder=1, aspect="equal", vmin=0, vmax=H.max())
    lw = dict(color="white", lw=1.0, zorder=2)
    ax.plot([0, PL, PL, 0, 0], [0, 0, PW, PW, 0], **lw); ax.plot([PL / 2] * 2, [0, PW], **lw)
    ax.add_patch(Circle((PL / 2, PW / 2), 9.15, fill=False, **lw))
    for x0 in (0, PL - 16.5): ax.add_patch(Rectangle((x0, (PW - 40.32) / 2), 16.5, 40.32, fill=False, **lw))
    ax.plot(r["_X"], r["_Y"], color="white", lw=0.9, alpha=0.8, zorder=3)
    ax.scatter([r["_X"][0]], [r["_Y"][0]], s=70, color="white", zorder=4); ax.annotate("start", (r["_X"][0], r["_Y"][0]), color="white", xytext=(5, 5), textcoords="offset points")
    ax.set_xlim(-3, PL + 3); ax.set_ylim(-3, PW + 3); ax.set_aspect("equal"); ax.set_title("where they were (heat) and the path", fontsize=10)
    ax.set_xlabel("metres along the pitch"); ax.set_ylabel("metres across")
    tx = fig.add_subplot(gs[0, 2]); tx.axis("off")
    team = {"W": "White team", "B": "Blue team"}.get(r["team"], "")
    rows = [("Team", team + (" (keeper)" if r["role"] == "GK" else "")), ("On screen", f"{r['seen_s']:.0f} s  ({r['estimated_pct']}% estimated)"),
            ("Distance", f"{r['distance_m']:.0f} m"), ("Average speed", f"{r['avg_speed_kmh']} km/h"), ("Top speed (1 s)", f"{r['top_speed_kmh']} km/h"),
            ("Sprints", f"{r['sprints']}"), ("Average position", f"{r['avg_x']:.0f} m along, {r['avg_y']:.0f} m across"),
            ("Time on the ball", f"{r['time_on_ball_s']} s"), ("Passes made / received", f"{r['passes_made']} / {r['passes_received']}")]
    for i, (a, b) in enumerate(rows):
        tx.text(0, 1 - i * 0.11, a, fontsize=9, color="#9aa", va="top"); tx.text(0, 1 - i * 0.11 - 0.045, b, fontsize=12, color="white", va="top", fontweight="bold")
    a1 = fig.add_subplot(gs[1, :2]); sp = r["_sp"] * 3.6
    a1.plot(r["_t"], np.where(np.isfinite(sp), np.minimum(sp, 36), np.nan), color="#ffd54a", lw=1.5)
    for y in (7, 14, 21): a1.axhline(y, color="#556", lw=0.6, ls="--")
    a1.set_ylabel("km/h"); a1.set_xlabel("seconds into the clip"); a1.set_title("speed (1-second average)", fontsize=10)
    a2 = fig.add_subplot(gs[1, 2]); vals = [r[f"{z}_pct"] for z in ZONES]
    a2.barh(ZONES, vals, color="#81c784"); a2.invert_yaxis(); a2.set_xlabel("% of time"); a2.set_title("speed zones", fontsize=10)
    fig.suptitle(f"{r['label']}", fontsize=18, x=0.07, ha="left", y=0.98)
    fig.text(0.5, 0.005, "Estimates: pitch calibration fitted by hand from the lines (1-3 m error near the camera, more at the far side). 30-second clip.", ha="center", fontsize=8, color="#889")
    fig.savefig(OUT / "players" / f"{r['label']}.png", dpi=80, facecolor=fig.get_facecolor()); plt.close(fig)
print("wrote", len(res), "pages in", OUT / "players")
