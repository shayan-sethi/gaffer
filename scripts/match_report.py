#!/usr/bin/env python3
"""
Everything a fixed-camera clip with identities, a ball track and touch-based passes supports.
  python match_report.py <out_dir> <video> <start_s> <seconds> [sfps=10]
Reads identities.json, ball_track.csv, pass_events.json, pass_spells.json. Writes into <out_dir>/report/:
  summary.json, player_metrics.csv, events.csv, pass_network.png, timeline.png, players/<label>.png, report.zip
Distances/speeds are ROUGH: picture pixels are scaled by each player's own box height (1.75 m), no pitch calibration, depth movement under-counted -> lower bounds.
"""
import sys, os, json, csv, math, zipfile
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import gaussian_filter

D, VIDEO, START, SECS = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
SFPS = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
OUT = D / "report"
(OUT / "players").mkdir(parents=True, exist_ok=True)
meta = json.load(open(D / "meta.json")); ids = json.load(open(D / "identities.json"))
ev = json.load(open(D / "pass_events.json")); spells = json.load(open(D / "pass_spells.json"))
TEAMNAME = {"W": os.environ.get("NAME_W", "Pink bibs"), "B": os.environ.get("NAME_B", "No bibs")}
TCOL = {"W": "#ff5fa8", "B": "#4aa8ff"}
trk = {}
for r in csv.DictReader(open(D / "ball_track.csv")):
    if r["x"] != "":
        trk[int(round(float(r["frame"])))] = (float(r["x"]), float(r["y"]), r["source"])
NR = len(open(D / "dets.jsonl").readlines())
H_M, MAXV = 1.75, 9.0
rows_frames = sorted(json.loads(l)["frame"] for l in open(D / "dets.jsonl"))
cap = cv2.VideoCapture(meta["video"]); cap.set(1, rows_frames[NR // 2]); ok, bg = cap.read()
Y0 = int(os.environ.get("CROP_Y0", 380)); bg = bg[Y0:, :]

lab = {I["label"]: I for I in ids.values()}
order = sorted(lab, key=lambda l: (lab[l]["team"] != "W", l))


def sm(v, w=5):
    p = np.pad(v, (w // 2, w // 2), mode="edge")
    return np.array([np.median(p[i:i + w]) for i in range(len(v))])


stats = {}
for l in order:
    I = lab[l]; fr = sorted(int(f) for f in I["rows"]); b = np.array([I["rows"][str(f)][:4] for f in fr])
    real = np.array([not I["rows"][str(f)][4] for f in fr])
    fx, fy, h = sm((b[:, 0] + b[:, 2]) / 2), sm(b[:, 3]), sm(b[:, 3] - b[:, 1], 9)
    t = np.array(fr) / SFPS
    step = np.hypot(np.diff(fx), np.diff(fy)) * H_M / np.maximum((h[1:] + h[:-1]) / 2, 1)
    dt = np.diff(t); v = step / np.maximum(dt, 1e-6)
    ok_ = (dt <= 0.25) & (v <= MAXV); dist = float(step[ok_].sum())
    sp = np.full(len(fr), np.nan)
    for i in range(len(v)):
        w = v[max(0, i - 9):i + 1]; okw = ok_[max(0, i - 9):i + 1]
        if okw.sum() >= 6:
            sp[i + 1] = float(np.mean(w[okw]))
    g = np.isfinite(sp)
    pc = lambda m_: round(100 * float(np.mean(m_))) if g.any() else None
    stats[l] = dict(label=l, team=I["team"], team_name=TEAMNAME[I["team"]], on_screen_pct=round(100 * len(fr) / NR),
                    estimated_pct=round(100 * (1 - float(real.mean()))), distance_m_rough=round(dist, 1),
                    avg_speed_kmh_rough=round(float(np.nanmean(sp)) * 3.6, 1) if g.any() else None,
                    top_speed_kmh_rough=round(min(float(np.nanmax(sp)) * 3.6, 32.0), 1) if g.any() else None,
                    time_standing_pct=pc(sp[g] < 0.5), time_walking_pct=pc((sp[g] >= 0.5) & (sp[g] < 1.94)),
                    time_jogging_pct=pc((sp[g] >= 1.94) & (sp[g] < 3.89)), time_running_pct=pc(sp[g] >= 3.89),
                    mean_x_px=round(float(fx.mean())), mean_y_px=round(float(fy.mean())),
                    spread_x_px=round(float(fx.std())), spread_y_px=round(float(fy.std())),
                    _fx=fx, _fy=fy, _t=t, _sp=sp)

for l in order:
    S_ = [s for s in spells if s["label"] == l]
    stats[l].update(
        possessions=len(S_), time_in_control_s=round(sum((s["last"] - s["first"]) / SFPS + 0.2 for s in S_), 1),
        passes_made=sum(1 for e in ev if e["frm"] == l and e["kind"] == "pass"),
        passes_received=sum(1 for e in ev if e["to"] == l and e["kind"] == "pass"),
        passes_made_seen=sum(1 for e in ev if e["frm"] == l and e["kind"] == "pass" and e["confidence"] == "seen"),
        lost_to_opponent=sum(1 for e in ev if e["frm"] == l and e["kind"] == "turnover"),
        won_from_opponent=sum(1 for e in ev if e["to"] == l and e["kind"] == "turnover"))

team = {}
for tm in ("W", "B"):
    L = [l for l in order if lab[l]["team"] == tm]
    team[tm] = dict(name=TEAMNAME[tm], players=len(L), passes=sum(stats[l]["passes_made"] for l in L),
                    passes_seen=sum(stats[l]["passes_made_seen"] for l in L), balls_lost=sum(stats[l]["lost_to_opponent"] for l in L),
                    balls_won=sum(stats[l]["won_from_opponent"] for l in L), possessions=sum(stats[l]["possessions"] for l in L),
                    time_in_control_s=round(sum(stats[l]["time_in_control_s"] for l in L), 1),
                    distance_m_rough=round(sum(stats[l]["distance_m_rough"] for l in L)),
                    shape_width_px=int(np.ptp([stats[l]["mean_x_px"] for l in L])), shape_depth_px=int(np.ptp([stats[l]["mean_y_px"] for l in L])))
tot = sum(team[t]["time_in_control_s"] for t in team) or 1
for t in team:
    team[t]["possession_pct_by_touches"] = round(100 * team[t]["time_in_control_s"] / tot)

ks = sorted(trk); seen = [k for k in ks if trk[k][2] == "det"]
pairs = [(a, b) for a, b in zip(ks, ks[1:]) if b - a <= 2 and trk[a][2] != "pred" and trk[b][2] != "pred"]
path = sum(math.hypot(trk[b][0] - trk[a][0], trk[b][1] - trk[a][1]) for a, b in pairs)
sp_b = [math.hypot(trk[b][0] - trk[a][0], trk[b][1] - trk[a][1]) / ((b - a) / SFPS) for a, b in pairs]
hid = [(a / SFPS, b / SFPS) for a, b in zip(ks, ks[1:]) if b - a > 2]
ball = dict(tracked_pct=round(100 * len(ks) / (NR / 2)), seen_by_detector_pct=round(100 * len(seen) / (NR / 2)), path_px=round(path),
            mean_speed_px_s=round(float(np.mean(sp_b))), max_speed_px_s=round(float(np.max(sp_b))),
            hidden_stretches_s=[(round(a, 1), round(b, 1)) for a, b in hid], longest_hidden_s=round(max([b - a for a, b in hid], default=0), 1))
pl = [e for e in ev if e["kind"] == "pass"]
summary = dict(clip=dict(video=os.path.basename(VIDEO), start_s=START, seconds=SECS), teams=team, ball=ball, events=len(ev),
               average_pass_ball_travel_px=round(float(np.mean([e["ball_travel_px"] for e in pl]))) if pl else None,
               longest_pass_px=max([e["ball_travel_px"] for e in pl], default=0),
               notes="distance/speed are rough (no pitch calibration); passes marked probable are not confirmed; only completed passes are visible")
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)

cols = [c for c in next(iter(stats.values())) if not c.startswith("_")]
with open(OUT / "player_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for l in order:
        w.writerow({c: stats[l][c] for c in cols})
with open(OUT / "events.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["time_in_clip_s", "clip_time", "kind", "from", "to", "confidence", "ball_hidden_s", "ball_travel_px"])
    for e in ev:
        t = START + e["row"] / SFPS
        w.writerow([round(e["row"] / SFPS, 1), f"{int(t // 60)}:{t % 60:04.1f}", e["kind"], e["frm"], e["to"], e["confidence"], e["gap_s"], e["ball_travel_px"]])

plt.style.use("dark_background")
# ---- pass network
fig, ax = plt.subplots(figsize=(13, 5.4)); ax.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB), alpha=0.55)
cnt = {}
for e in pl:
    cnt[(e["frm"], e["to"])] = cnt.get((e["frm"], e["to"]), 0) + 1
pos = {l: (stats[l]["mean_x_px"], stats[l]["mean_y_px"] - Y0) for l in order}
for (a, b), c in cnt.items():
    ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle="-|>", mutation_scale=18 + 6 * c, lw=1.5 + 2.2 * c, color="#ffd54a",
                                 alpha=0.85, connectionstyle="arc3,rad=0.25", shrinkA=16, shrinkB=16))
    ax.text((pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2 - 18, str(c), color="white", fontsize=11, ha="center", weight="bold")
for l in order:
    ax.scatter(*pos[l], s=520, color=TCOL[lab[l]["team"]], edgecolor="white", zorder=4)
    ax.text(pos[l][0], pos[l][1], l, ha="center", va="center", fontsize=9, color="black", weight="bold", zorder=5)
ax.set_title("Pass network: arrows = completed passes (number = how many); circles = average position in the camera view", fontsize=11); ax.axis("off")
fig.savefig(OUT / "pass_network.png", dpi=90, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close(fig)

# ---- timeline
fig, axs = plt.subplots(2, 1, figsize=(14, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[1.0, 1.7]))
tt = np.array(ks) / SFPS
spd = [np.nan] + [math.hypot(trk[b][0] - trk[a][0], trk[b][1] - trk[a][1]) / ((b - a) / SFPS) if b - a <= 2 else np.nan for a, b in zip(ks, ks[1:])]
axs[0].plot(tt, spd, color="#ffd54a", lw=1.4); axs[0].set_ylabel("ball speed (px/s)")
for a, b in hid:
    axs[0].axvspan(a, b, color="#888", alpha=0.35)
for e in ev:
    axs[0].axvline(e["row"] / SFPS, color="#66ff99", lw=1, alpha=0.7)
axs[0].set_title("Ball speed (grey = ball hidden); green lines = passes", fontsize=10)
yl = {l: i for i, l in enumerate(order)}
for s in spells:
    axs[1].barh(yl[s["label"]], (s["last"] - s["first"]) / SFPS + 0.2, left=s["first"] / SFPS, color=TCOL[s["team"]], height=0.7)
for e in ev:
    axs[1].annotate("", xy=(e["row"] / SFPS, yl[e["to"]]), xytext=(e["frm_row"] / SFPS, yl[e["frm"]]),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.0 if e["confidence"] == "probable" else 1.9, ls="--" if e["confidence"] == "probable" else "-"))
axs[1].set_yticks(list(yl.values())); axs[1].set_yticklabels(list(yl)); axs[1].set_xlabel("seconds into the clip")
axs[1].set_title("Who is in control of the ball (bars) and passes (arrows; dashed = probable)", fontsize=10); axs[1].invert_yaxis()
fig.tight_layout(); fig.savefig(OUT / "timeline.png", dpi=90, facecolor=fig.get_facecolor()); plt.close(fig)

# ---- player pages
for l in order:
    S = stats[l]
    fig = plt.figure(figsize=(12.5, 7.2)); gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.2], width_ratios=[2.3, 1, 1], hspace=0.38, wspace=0.28)
    ax = fig.add_subplot(gs[0, :2]); ax.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB), alpha=0.7)
    hm = np.zeros(bg.shape[:2], np.float32)
    for x, y in zip(S["_fx"], S["_fy"] - Y0):
        if 0 <= int(y) < hm.shape[0] and 0 <= int(x) < hm.shape[1]:
            hm[int(y), int(x)] += 1
    hm = gaussian_filter(hm, 18)
    if hm.max() > 0:
        ax.imshow(np.ma.masked_less(hm / hm.max(), 0.06), cmap="YlOrRd", alpha=0.6, vmin=0, vmax=1)
    ax.plot(S["_fx"], S["_fy"] - Y0, color="white", lw=1.0, alpha=0.8)
    for e in pl:
        if e["frm"] == l and e["frm_row"] in trk:
            ax.scatter(trk[e["frm_row"]][0], trk[e["frm_row"]][1] - Y0, marker="^", s=90, color="#66ff99", zorder=5)
        if e["to"] == l and e["row"] in trk:
            ax.scatter(trk[e["row"]][0], trk[e["row"]][1] - Y0, marker="v", s=90, color="#ffd54a", zorder=5)
    ax.set_title(f"{l} ({S['team_name']}): path and where they spent time. Green ^ = passed from here, yellow v = received here", fontsize=10); ax.axis("off")
    tx = fig.add_subplot(gs[0, 2]); tx.axis("off")
    rows_ = [("On screen", f"{S['on_screen_pct']}%  ({S['estimated_pct']}% estimated)"), ("Distance (rough)", f"{S['distance_m_rough']} m"),
             ("Avg / top speed (rough)", f"{S['avg_speed_kmh_rough']} / {S['top_speed_kmh_rough']} km/h"),
             ("Possessions / time in control", f"{S['possessions']} / {S['time_in_control_s']} s"),
             ("Passes made / received", f"{S['passes_made']} / {S['passes_received']}"),
             ("Ball lost / won", f"{S['lost_to_opponent']} / {S['won_from_opponent']}"),
             ("Stand / walk / jog / run", f"{S['time_standing_pct']} / {S['time_walking_pct']} / {S['time_jogging_pct']} / {S['time_running_pct']} %")]
    for i, (a, b) in enumerate(rows_):
        tx.text(0, 1 - i * 0.14, a, fontsize=9, color="#9aa", va="top"); tx.text(0, 1 - i * 0.14 - 0.055, b, fontsize=12, color="white", va="top", weight="bold")
    a1 = fig.add_subplot(gs[1, :2]); spk = S["_sp"] * 3.6
    a1.plot(S["_t"], np.where(np.isfinite(spk), np.minimum(spk, 32), np.nan), color="#ffd54a", lw=1.4)
    for s in spells:
        if s["label"] == l:
            a1.axvspan(s["first"] / SFPS, s["last"] / SFPS + 0.2, color=TCOL[S["team"]], alpha=0.35)
    a1.set_ylabel("km/h (rough)"); a1.set_xlabel("seconds into the clip"); a1.set_title("speed; shaded = player had the ball", fontsize=9)
    a2 = fig.add_subplot(gs[1, 2]); nm = ["standing", "walking", "jogging", "running"]
    a2.barh(nm, [S["time_standing_pct"] or 0, S["time_walking_pct"] or 0, S["time_jogging_pct"] or 0, S["time_running_pct"] or 0], color="#81c784"); a2.invert_yaxis(); a2.set_title("time by speed", fontsize=9)
    fig.text(0.5, 0.005, "Rough numbers: no pitch calibration (steps scaled by the player's own height, depth movement under-counted). Passes marked probable are not confirmed. 30-second clip.",
             ha="center", fontsize=8, color="#889")
    fig.savefig(OUT / "players" / f"{l}.png", dpi=75, facecolor=fig.get_facecolor()); plt.close(fig)

z = zipfile.ZipFile(OUT / "report.zip", "w", zipfile.ZIP_DEFLATED)
for p in sorted(OUT.rglob("*")):
    if p.is_file() and p.name != "report.zip":
        z.write(p, p.relative_to(OUT))
z.close()
print(json.dumps(summary["teams"], indent=1))
print("ball:", json.dumps({k: ball[k] for k in ("tracked_pct", "seen_by_detector_pct", "path_px", "mean_speed_px_s", "max_speed_px_s", "longest_hidden_s")}))
print("wrote", OUT)
