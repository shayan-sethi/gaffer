#!/usr/bin/env python3
"""
Lock a fixed-camera clip to N individual player identities (no teams, no ball, no kit colours).

  python clip_ids.py <video> <out_dir> [start_s=60] [seconds=60] [n_players=14] [static_probe.json]

Stage 1  detect (resumable, appends to dets.jsonl): COCO yolov8n person boxes + torso colour.
Stage 2  analyse: drop advert-board figures (static spots) and people whose feet are off the turf
         (scripts/pitch_mask.py), ByteTrack, then chain the tracklets into exactly N identities:
           - link tracklet ends to later tracklet starts (time gap, distance in player heights, colour)
           - keep the N longest chains as identities
           - hang the leftover tracklets on whichever identity they fit (no time overlap, close in space)
           - what still doesn't fit is reported as "unassigned" and drawn thin grey
Outputs: summary.json, ids.webm (boxed video with #1..#N labels).
"""
import sys, os, json, time, colorsys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from pitch_mask import build_mask, feet_on_pitch, is_static

VIDEO, OUT = sys.argv[1], Path(sys.argv[2])
START = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
SECONDS = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
N = int(sys.argv[5]) if len(sys.argv) > 5 else 14
PROBE = sys.argv[6] if len(sys.argv) > 6 else None
STRIDE = 2
IMGSZ, CONF = 1920, 0.15
MIN_H = 18            # px; far players on an end-on view are ~25 px tall
MIN_TRACK = 6         # sampled frames
START_FRAMES = int(os.environ.get('START_FRAMES', 120))
EXTRA = int(os.environ.get('EXTRA_SLOTS', 0))              # identities beyond N, for officials   # min frames that justify opening a new identity


def torso_rgb(frame, b):
    x1, y1, x2, y2 = [int(v) for v in b]
    h, w = y2 - y1, x2 - x1
    crop = frame[max(y1 + int(.2 * h), 0):y1 + int(.55 * h), x1 + int(.2 * w):x2 - int(.2 * w)]
    if crop.size == 0:
        return [0, 0, 0]
    m = crop.reshape(-1, 3).mean(0)
    return [round(float(m[2])), round(float(m[1])), round(float(m[0]))]


# ------------------------------------------------------------------ stage 1
def detect():
    from ultralytics import YOLO
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "dets.jsonl"
    done = {json.loads(l)["frame"] for l in open(cache)} if cache.exists() else set()
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    f0 = int(round(START * fps))
    frames = list(range(f0, f0 + int(SECONDS * fps), STRIDE))
    (OUT / "meta.json").write_text(json.dumps({"video": VIDEO, "fps": fps, "stride": STRIDE, "start_frame": f0,
                                               "n": N, "width": int(cap.get(3)), "height": int(cap.get(4))}))
    model = YOLO("yolov8n.pt")
    t0, k = time.time(), 0
    todo = [f for f in frames if f not in done]
    with open(cache, "a") as out:
        for fi in todo:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                break
            r = model(frame, imgsz=IMGSZ, conf=CONF, classes=[0], verbose=False)[0]
            dets = [[*[round(float(v), 1) for v in b], round(float(c), 3), *torso_rgb(frame, b)]
                    for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
            out.write(json.dumps({"frame": fi, "dets": dets}) + "\n")
            out.flush()
            k += 1
            if k % 10 == 1:
                el = time.time() - t0
                print(f"{k}/{len(todo)} {el / k:.2f}s/frame eta {el / k * (len(todo) - k) / 60:.1f} min", flush=True)
    print("detect done", flush=True)


# ------------------------------------------------------------------ stage 2
def pitch_mask_for_window(meta):
    cap = cv2.VideoCapture(VIDEO)
    fr = []
    for i in range(6):
        cap.set(cv2.CAP_PROP_POS_FRAMES, meta["start_frame"] + int(i * SECONDS * meta["fps"] / 6))
        ok, f = cap.read()
        if ok:
            fr.append(f)
    return build_mask(fr)


def drop_duplicate_boxes(a):
    """Two boxes on one person (different sizes, so NMS keeps both): same feet position, one mostly inside
    the other. Keep the more confident one. Boxes on genuinely different people differ in foot depth."""
    if len(a) < 2:
        return a
    keep = np.ones(len(a), bool)
    order = np.argsort(-a[:, 4])
    for ii, i in enumerate(order):
        if not keep[i]:
            continue
        for j in order[ii + 1:]:
            if not keep[j]:
                continue
            x1, y1, x2, y2 = a[i, :4]
            u1, v1, u2, v2 = a[j, :4]
            iw, ih = min(x2, u2) - max(x1, u1), min(y2, v2) - max(y1, v1)
            if iw <= 0 or ih <= 0:
                continue
            small = min((x2 - x1) * (y2 - y1), (u2 - u1) * (v2 - v1))
            hs = min(y2 - y1, v2 - v1)
            same_feet = abs(y2 - v2) < 0.25 * hs and abs((x1 + x2) / 2 - (u1 + u2) / 2) < 0.5 * max(x2 - x1, u2 - u1)
            iou = iw * ih / ((x2 - x1) * (y2 - y1) + (u2 - u1) * (v2 - v1) - iw * ih)
            if iou > 0.45 and same_feet:             # IoU, not just containment: a player standing partly behind another is NOT a duplicate
                keep[j] = False
    return a[keep]


def track(rows, mask, spots, sfps):
    import supervision as sv
    tr = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=int(2 * sfps),
                      minimum_matching_threshold=0.8, frame_rate=max(1, int(sfps)))
    per_frame, dropped = [], {"static": 0, "off_pitch": 0, "small": 0}
    for r in rows:
        keep = []
        for d in r["dets"]:
            b = d[:4]
            if b[3] - b[1] < MIN_H:
                dropped["small"] += 1
            elif is_static(b, spots):
                dropped["static"] += 1
            elif not feet_on_pitch(mask, b):
                dropped["off_pitch"] += 1
            else:
                keep.append(d)
        if keep:
            a = drop_duplicate_boxes(np.array(keep, np.float32))
            det = sv.Detections(xyxy=a[:, :4], confidence=a[:, 4], class_id=np.zeros(len(a), int))
            det = det.with_nms(threshold=0.7, class_agnostic=True)
            upd = tr.update_with_detections(det)
        else:
            a = np.zeros((0, 8), np.float32)
            upd = tr.update_with_detections(sv.Detections.empty())
        out = []
        for tid, box, cf in zip(upd.tracker_id, upd.xyxy, upd.confidence):
            j = int(np.argmin(np.abs(a[:, :4] - box).sum(1))) if len(a) else None
            out.append(dict(tid=int(tid), box=[float(v) for v in box], conf=float(cf),
                            rgb=[float(v) for v in a[j, 5:8]] if j is not None else [0, 0, 0],
                            lab=([float(v) for v in a[j, 8:11]] if (j is not None and a.shape[1] >= 11 and a[j, 8] > -900) else None)))
        per_frame.append(out)
    return per_frame, dropped


def foot(o):
    return np.array([(o["box"][0] + o["box"][2]) / 2, o["box"][3]])


def height(o):
    return o["box"][3] - o["box"][1]


def col_dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)) / 441.7)


APP_W = float(os.environ.get("APP_W", 8))     # weight of the appearance-embedding term in link costs


def link_cost(a_end, b_start, gap_s, ea=None, ec=None):
    """cost of saying 'the person at a_end (a frame) later reappears at b_start'; None if impossible.
    ea / ec: optional appearance embeddings (unit vectors) of the two tracklets - a soft extra cost, never a veto."""
    h = max(np.median([height(o) for o in a_end[-5:]]), np.median([height(o) for o in b_start[:5]]))
    dist = float(np.linalg.norm(foot(a_end[-1]) - foot(b_start[0])))
    allow = h * (0.6 + 2.0 * gap_s)
    cd = col_dist(np.median([o["rgb"] for o in a_end[-8:]], 0), np.median([o["rgb"] for o in b_start[:8]], 0))
    if dist > allow or cd > 0.40:
        return None
    cost = gap_s + dist / h / 2 + 3 * cd
    if ea is not None and ec is not None:
        cost += APP_W * (1.0 - float(ea @ ec))
    return cost


def split_on_shirt_change(tracks, win=9, min_run=8):
    """ByteTrack sometimes carries one ID from one player to another when they overlap. If the torso
    colour of a tracklet flips between light and dark (and stays flipped for >= min_run samples) it is two
    different people: cut the tracklet there, so no identity can ever contain both a light and a dark shirt."""
    allb = []
    sm = {}
    for tid, seq in tracks.items():
        b = np.array([np.mean(o["rgb"]) for _, o in seq])
        m = np.array([np.median(b[max(0, i - win // 2):i + win // 2 + 1]) for i in range(len(b))])
        sm[tid] = m
        allb.extend(m.tolist())
    allb = np.array(allb)
    lo, hi = float(np.percentile(allb, 15)), float(np.percentile(allb, 85))
    for _ in range(30):
        thr = (lo + hi) / 2
        lo = float(allb[allb <= thr].mean()) if (allb <= thr).any() else lo
        hi = float(allb[allb > thr].mean()) if (allb > thr).any() else hi
    thr = (lo + hi) / 2
    out = {}
    for tid, seq in tracks.items():
        cls = (sm[tid] > thr).astype(int)
        # run-length smooth: a class run shorter than min_run is absorbed by its neighbours
        runs, start = [], 0
        for i in range(1, len(cls) + 1):
            if i == len(cls) or cls[i] != cls[start]:
                runs.append([start, i, int(cls[start])])
                start = i
        merged = []
        for r in runs:
            if merged and (r[1] - r[0] < min_run or merged[-1][2] == r[2]):
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        if len(merged) > 1 and merged[0][1] - merged[0][0] < min_run:
            merged[1][0] = merged[0][0]
            merged = merged[1:]
        for j, (a, b2, _) in enumerate(merged):
            out[tid * 1000 + j] = seq[a:b2]
    return out


def lock_identities_greedy(per_frame, sfps):
    tracks = {}
    for fi, out in enumerate(per_frame):
        for o in out:
            tracks.setdefault(o["tid"], []).append((fi, o))
    tracks = split_on_shirt_change(tracks)
    tl = {t: s for t, s in tracks.items() if len(s) >= MIN_TRACK}
    ids = sorted(tl, key=lambda t: tl[t][0][0])
    # teams per tracklet: lighter / darker shirt (2-means on median torso brightness, length-weighted).
    # A link between a light and a dark tracklet is never allowed, so a chain can't mix teams.
    tb = {t: float(np.median([np.mean(o["rgb"]) for _, o in tl[t]])) for t in ids}
    w = np.array([len(tl[t]) for t in ids], float)
    v = np.array([tb[t] for t in ids])
    lo, hi = float(np.percentile(v, 15)), float(np.percentile(v, 85))
    for _ in range(30):
        thr = (lo + hi) / 2
        lo = float(np.average(v[v <= thr], weights=w[v <= thr])) if (v <= thr).any() else lo
        hi = float(np.average(v[v > thr], weights=w[v > thr])) if (v > thr).any() else hi
    thr = (lo + hi) / 2
    team_of = {t: ("W" if tb[t] > thr else "B") for t in ids}
    # 1) chain tracklets
    cands = []
    for a in ids:
        for c in ids:
            if a == c or team_of[a] != team_of[c]:
                continue
            gap = tl[c][0][0] - tl[a][-1][0]
            if 1 <= gap <= int(3 * sfps):
                cost = link_cost([o for _, o in tl[a]], [o for _, o in tl[c]], gap / sfps)
                if cost is not None:
                    cands.append((cost, a, c))
    nxt, prv = {}, {}
    for _, a, c in sorted(cands):
        if a in nxt or c in prv:
            continue
        nxt[a], prv[c] = c, a
    chains = []
    for i in ids:
        if i in prv:
            continue
        ch, cur = [i], i
        while cur in nxt:
            cur = nxt[cur]
            ch.append(cur)
        chains.append(ch)

    def frames_of(ch):
        d = {}
        for t in ch:
            for fi, o in tl[t]:
                d[fi] = o
        return d

    cf = [(frames_of(ch), ch) for ch in chains]
    cf.sort(key=lambda x: -len(x[0]))
    # 2) longest chain first: a chain joins the identity it fits (no time overlap, close in space,
    #    similar colour) and only opens a new identity if it fits none and fewer than N exist.
    #    So a person's second chain never uses up a slot that another player needs.
    ident, unassigned = [], []
    cap_team = {"W": N // 2, "B": N - N // 2}
    for f, ch in cf:
        fl = sorted(f)
        cteam = team_of[ch[0]]
        best, best_cost = None, None
        for k, I in enumerate(ident):
            if I["team"] != cteam or set(f) & set(I["frames"]):
                continue
            ifr = np.array(sorted(I["frames"]))
            # compare each end of the leftover with the nearest identity frame in time
            costs = []
            for end_fi, is_start in ((fl[0], True), (fl[-1], False)):
                j = int(np.searchsorted(ifr, end_fi))
                for jj in (j - 1, j):
                    if 0 <= jj < len(ifr):
                        gap = abs(int(ifr[jj]) - end_fi)
                        if 1 <= gap <= int(6 * sfps):
                            oa, ob = I["frames"][int(ifr[jj])], f[end_fi]
                            first, second = ((oa, ob) if ifr[jj] < end_fi else (ob, oa))
                            c = link_cost([first], [second], gap / sfps)
                            if c is not None:
                                costs.append(c)
            if costs and (best_cost is None or min(costs) < best_cost):
                best, best_cost = k, min(costs)
        if best is None:
            if sum(1 for I in ident if I["team"] == cteam) < cap_team[cteam]:
                ident.append(dict(frames=dict(f), tids=list(ch), team=cteam))
            else:
                unassigned.append((f, ch))
        else:
            ident[best]["frames"].update(f)
            ident[best]["tids"] += ch
    # 3) fill short gaps (<= 1 s) by interpolating the box, so a brief occlusion doesn't blank the player
    for I in ident:
        fr = sorted(I["frames"])
        for a, b in zip(fr, fr[1:]):
            if 2 <= b - a <= int(1.0 * sfps) + 1:
                oa, ob = I["frames"][a], I["frames"][b]
                for f in range(a + 1, b):
                    t = (f - a) / (b - a)
                    I["frames"][f] = dict(tid=oa["tid"], conf=0.0, rgb=oa["rgb"], interp=True,
                                          box=[x + t * (y - x) for x, y in zip(oa["box"], ob["box"])])
    return ident, unassigned, len(tracks), len(tl), len(chains)


def lab_feature(rgb):
    px = np.uint8([[[min(255, max(0, round(float(v)))) for v in rgb[::-1]]]])      # RGB -> BGR for OpenCV
    L, a, b = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    return np.array([L * 100 / 255, 1.5 * (a - 128), 1.5 * (b - 128)])


def flag_officials(ident, class_centre):
    """With EXTRA spare slots there are more identities than players. The EXTRA identities whose shirt colour
    is furthest from their own team's colour (relative to how spread that team is) are marked as officials."""
    if EXTRA <= 0 or len(ident) <= EXTRA or not os.environ.get("OFFICIALS_BY_COLOUR"):
        return          # colour alone cannot tell officials from players (it flagged a white player): use position instead
    d = {}
    for cls in ("W", "B"):
        mem = [I for I in ident if I["cls"] == cls]
        dist = [np.linalg.norm(I["feat"] - class_centre[cls]) for I in mem]
        scale = max(float(np.median(dist)), 1e-6)
        for I, x in zip(mem, dist):
            I["outlier"] = float(x / scale)
    for I in sorted(ident, key=lambda I: -I.get("outlier", 0))[:EXTRA]:
        I["official"] = True


def assign_roles(ident):
    """Colour class -> team, and find the keepers. Teams are N/2 each. If a colour class has too many identities
    (a keeper wearing the other team's colours, or a bib), the extra ones are keepers at the pitch ends; the
    surplus identity that goes to the other team is a GUESS (flagged), because shirt colour can't tell whose goalkeeper it is."""
    for I in ident:
        I["mean_x"] = float(np.mean([foot(o)[0] for o in I["frames"].values()]))
        I["role"], I["team"], I["guess"] = ("OFFICIAL", "-", False) if I.get("official") else ("P", I["cls"], False)
        if I["cls"] == "X":
            I["team"], I["role"] = "?", "ODD"          # odd shirt colour: keeper or referee, decided by position later
    ident_all = ident
    ident = [I for I in ident_all if not I.get("official") and I["cls"] in ("W", "B")]
    n = len(ident)
    big, small = ("W", "B") if sum(I["cls"] == "W" for I in ident) >= sum(I["cls"] == "B" for I in ident) else ("B", "W")
    nb = sum(I["cls"] == big for I in ident)
    extra = nb - n // 2
    if extra > 0:
        pool = sorted([I for I in ident if I["cls"] == big], key=lambda I: I["mean_x"])
        ends = []
        for k in range(extra):                         # alternate left / right extremes
            ends.append(pool.pop(0) if k % 2 == 0 else pool.pop(-1))
        small_x = float(np.mean([I["mean_x"] for I in ident if I["cls"] == small] or [0]))
        for I in ends:                                 # the keeper(s) closest to the small team's usual position defend its goal
            I["role"] = "GK"
        mid = sorted(ends, key=lambda I: abs(I["mean_x"] - small_x))
        for I in mid[:extra]:
            I["team"], I["guess"] = small, True
        # the other extreme of the big class is that team's own keeper
        if pool:
            far = pool.pop(0) if ends and ends[0]["mean_x"] > pool[0]["mean_x"] else pool.pop(-1)
            far["role"] = "GK"


class SigEmbedder:
    """Appearance embedding (SigLIP image features, cached locally) of a stretch of track: the mean over up to
    `per` crops of the whole player, spread over the stretch. Weak on its own (about 70% teammate matching),
    useful as one soft cue next to position, speed and shirt colour."""

    def __init__(self, video, rows, per=6):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch, self.rows, self.per = torch, rows, per
        self.proc = AutoProcessor.from_pretrained("google/siglip-base-patch16-224", local_files_only=True)
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224", local_files_only=True).eval()
        self.cap = cv2.VideoCapture(video)

    def __call__(self, seq):
        good = [(fi, o) for fi, o in seq if not o.get("interp") and (o["box"][3] - o["box"][1]) >= 40]
        if not good:
            return None
        pick = [good[i] for i in np.linspace(0, len(good) - 1, min(self.per, len(good))).astype(int)]
        crops = []
        for fi, o in pick:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.rows[fi]["frame"])
            ok, im = self.cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = [int(v) for v in o["box"]]
            c = im[max(0, y1):y2, max(0, x1):x2]
            if c.size:
                crops.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
        if not crops:
            return None
        with self.torch.no_grad():
            e = self.model.get_image_features(**self.proc(images=crops, return_tensors="pt"))
            e = self.torch.nn.functional.normalize(e, dim=-1).numpy()
        m = e.mean(axis=0)
        return m / np.linalg.norm(m)


def reteam(video, rows, ident, per=12):
    """Re-decide the teams of locked identities from the cached grass-free shirt colours (o["lab"]).
    Light vs dark/blue = the two biggest of 3 colour groups; a small odd group (keepers / officials) gets '?'."""
    cols = {}
    for k, I in enumerate(ident):
        if I.get("official"):
            continue
        v = [o["lab"] for o in I["frames"].values() if o.get("lab") is not None and not o.get("interp")]
        if len(v) >= 3:
            cols[k] = np.median(np.array(v), axis=0)
    ks = list(cols)
    X = np.array([cols[k] for k in ks], np.float32)
    cv2.setRNGSeed(0)
    _, lab_, cen = cv2.kmeans(X, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4), 20, cv2.KMEANS_PP_CENTERS)
    lab_ = lab_.ravel()
    sizes = np.bincount(lab_, minlength=3)
    team_ids = sorted(np.argsort(-sizes)[:2], key=lambda c: -cen[c][0])      # lighter first
    name = {int(team_ids[0]): "W", int(team_ids[1]): "B"}
    for k, l_ in zip(ks, lab_):
        ident[k]["team"] = name.get(int(l_), "?")
        ident[k]["cls"] = ident[k]["team"]
    return {k: cols[k] for k in ks}


class TrackletColour:
    """Grass-free shirt colour (Lab feature) of a stretch of track = median of the per-detection colours cached by
    clip_colours.py (o["lab"]). No video access, so it is instant. Returns None if there is none."""

    def __init__(self, video=None, rows=None, per=None):
        pass

    def __call__(self, seq):
        v = [o["lab"] for _, o in seq if o.get("lab") is not None and not o.get("interp")]
        return np.median(np.array(v), axis=0) if len(v) >= 2 else None


def lock_identities(per_frame, sfps, embedder=None, colourer=None):
    """Exact assignment instead of greedy matching.

    Per team, find K = N/2 non-overlapping chains of tracklets that together cover the most frames,
    where chaining tracklet a -> c costs link_cost(a, c). This is a min-cost flow on a DAG
    (the classic multi-object-tracking formulation), so one early wrong link can no longer block the
    right ones later. Tracklets on no path are returned as `unassigned` (bystanders, extra people).
    """
    import networkx as nx
    tracks = {}
    for fi, out in enumerate(per_frame):
        for o in out:
            tracks.setdefault(o["tid"], []).append((fi, o))
    tracks = split_on_shirt_change(tracks)
    tl = {t: s for t, s in tracks.items() if len(s) >= MIN_TRACK}
    ids = sorted(tl, key=lambda t: tl[t][0][0])
    # team class of each tracklet: length-weighted 2-means on median torso colour in Lab (lightness + chroma),
    # so blue-vs-white and black-vs-white both separate (brightness alone fails for blue under cloud)
    feat = {}
    for t in ids:
        f_ = colourer(tl[t]) if colourer is not None else None
        feat[t] = f_ if f_ is not None else lab_feature(np.median([o["rgb"] for _, o in tl[t]], axis=0))
    X = np.array([feat[t] for t in ids], np.float32)
    cv2.setRNGSeed(0)
    _, lab_, cen = cv2.kmeans(X, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4), 20, cv2.KMEANS_PP_CENTERS)
    lab_ = lab_.ravel()
    wts = np.array([len(tl[t]) for t in ids], float)
    size_ = np.array([wts[lab_ == c].sum() for c in range(3)])
    two = sorted(np.argsort(-size_)[:2], key=lambda c: -cen[c][0])           # lighter first
    nm = {int(two[0]): "W", int(two[1]): "B"}
    team_of = {t: nm.get(int(l_), "X") for t, l_ in zip(ids, lab_)}           # X = odd colour (keeper, referee)
    class_centre = {"W": cen[two[0]], "B": cen[two[1]], "X": cen[[c for c in range(3) if c not in two][0]]}
    tracklet_emb = {t: embedder(tl[t]) for t in ids} if embedder is not None else {}
    tracklet_feat = feat
    FRAME_REWARD, LINK_W, MAX_GAP = 100, 400, int(8 * sfps)
    START_COST = FRAME_REWARD * START_FRAMES      # a new identity must be worth at least this many frames
    # one flow for everybody: tracklets only chain within their own colour class, and the solver decides
    # how the N identities split between the classes (a keeper in a different colour makes it uneven, e.g. 4+6)
    G = nx.DiGraph()
    K = N + EXTRA                                       # spare slots for officials / people who are not players
    G.add_node("S", demand=-K)
    G.add_node("T", demand=K)
    G.add_edge("S", "T", capacity=K, weight=0)
    for t in ids:
        G.add_edge(("in", t), ("out", t), capacity=1, weight=-FRAME_REWARD * len(tl[t]))
        G.add_edge("S", ("in", t), capacity=1, weight=START_COST)
        G.add_edge(("out", t), "T", capacity=1, weight=0)
    for a_ in ids:
        for c_ in ids:
            if a_ == c_ or team_of[a_] != team_of[c_]:
                continue
            gap = tl[c_][0][0] - tl[a_][-1][0]
            if 1 <= gap <= MAX_GAP:
                cost = link_cost([o for _, o in tl[a_]], [o for _, o in tl[c_]], gap / sfps,
                                 tracklet_emb.get(a_), tracklet_emb.get(c_))
                if cost is not None:
                    G.add_edge(("out", a_), ("in", c_), capacity=1, weight=int(LINK_W * cost))
    flow = nx.min_cost_flow(G)
    nxt = {}
    for a_ in ids:
        for node, f in flow[("out", a_)].items():
            if f and node != "T":
                nxt[a_] = node[1]
    ident, used = [], set()
    for t in [t for t in ids if flow["S"].get(("in", t), 0)]:
        ch, cur = [t], t
        while cur in nxt:
            cur = nxt[cur]
            ch.append(cur)
        frames = {}
        for tt in ch:
            for fi, o in tl[tt]:
                frames[fi] = o
            used.add(tt)
        ident.append(dict(frames=frames, tids=ch, cls=team_of[ch[0]],
                          feat=np.average([tracklet_feat[tt] for tt in ch], axis=0, weights=[len(tl[tt]) for tt in ch])))
    flag_officials(ident, class_centre)
    assign_roles(ident)
    unassigned = [(dict((fi, o) for fi, o in tl[t]), [t]) for t in ids if t not in used]
    # fill short gaps (<= 1 s) by interpolating the box, so a brief occlusion doesn't blank the player
    for I in ident:
        fr = sorted(I["frames"])
        for a, b in zip(fr, fr[1:]):
            if 2 <= b - a <= int(1.0 * sfps) + 1:
                oa, ob = I["frames"][a], I["frames"][b]
                for f in range(a + 1, b):
                    t = (f - a) / (b - a)
                    I["frames"][f] = dict(tid=oa["tid"], conf=0.0, rgb=oa["rgb"], interp=True,
                                          box=[x + t * (y - x) for x, y in zip(oa["box"], ob["box"])])
    ident.sort(key=lambda I: (I["team"], I["role"] != "GK", -len(I["frames"])))
    return ident, unassigned, len(tracks), len(tl), len(ident)


# ------------------------------------------------------------------ render
def palette(n):
    return [tuple(int(255 * v) for v in colorsys.hsv_to_rgb(i / n, 0.85, 1.0))[::-1] for i in range(n)]


def render(meta, per_frame, rows, ident, unassigned, sfps):
    cap = cv2.VideoCapture(VIDEO)
    y0 = int(meta.get("band_y0", 0))                    # panoramas: drop the sky, keep the pitch band
    W = 3200 if meta["width"] / meta["height"] > 2.5 else 1280
    H = int(W * (meta["height"] - y0) / meta["width"])
    w = cv2.VideoWriter(str(OUT / "ids.webm"), cv2.VideoWriter_fourcc(*"VP80"), sfps, (W, H))
    tcol = {"W": (255, 255, 255), "B": (0, 140, 255)}
    cols = [tcol.get(I.get("team"), (0, 255, 0)) for I in ident]
    cnt, labels = {}, []
    for I in ident:
        cnt[I.get("team")] = cnt.get(I.get("team"), 0) + 1
        labels.append(f"{I.get('team', '')}{cnt[I.get('team')]}")
    sc = W / meta["width"]
    unass = {fi: [] for fi in range(len(rows))}
    for f, _ in unassigned:
        for fi, o in f.items():
            unass[fi].append(o)
    for fi, r in enumerate(rows):
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame[y0:], (W, H), interpolation=cv2.INTER_AREA)
        for o in unass[fi]:
            b = [int(v * sc) for v in (o["box"][0], o["box"][1] - y0, o["box"][2], o["box"][3] - y0)]
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (160, 160, 160), 1)
        vis = 0
        for k, I in enumerate(ident):
            o = I["frames"].get(fi)
            if o is None:
                continue
            vis += 1
            b = [int(v * sc) for v in (o["box"][0], o["box"][1] - y0, o["box"][2], o["box"][3] - y0)]
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), cols[k], 1 if o.get('interp') else 2)
            cv2.putText(frame, labels[k], (b[0], max(14, b[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cols[k], 2)
        cv2.rectangle(frame, (0, 0), (520, 44), (0, 0, 0), -1)
        cv2.putText(frame, f"t={(r['frame'] - meta['start_frame']) / meta['fps'] + START:5.1f}s  identified {vis}/{len(ident)}",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        w.write(frame)
    w.release()


def analyse():
    meta = json.load(open(OUT / "meta.json"))
    rows = sorted((json.loads(l) for l in open(OUT / "dets.jsonl")), key=lambda r: r["frame"])
    sfps = meta["fps"] / meta["stride"]
    spots = []
    if PROBE:
        import pitch_mask
        spots = pitch_mask.static_spots(json.load(open(PROBE)))
    print("static spots", [(round(x), round(y)) for x, y in spots])
    mask = pitch_mask_for_window(meta)
    if mask.mean() < 0.15:     # dark / grey turf (e.g. Tolworth at dusk) fails the green test
        print(f"WARNING: turf mask only covers {100 * mask.mean():.0f}% of the frame - not using it", flush=True)
        mask = np.ones_like(mask)
    per_frame, dropped = track(rows, mask, spots, sfps)
    ident, unassigned, n_raw, n_long, n_chains = lock_identities(per_frame, sfps)
    nfr = len(rows)
    vis_each = [len(I["frames"]) / nfr for I in ident]
    counts = [sum(1 for I in ident if fi in I["frames"]) for fi in range(nfr)]
    summary = dict(
        seconds=SECONDS, sampled_frames=nfr, target_players=N, dropped_detections=dropped,
        raw_tracker_ids=n_raw, tracklets_ge_min=n_long, chains=n_chains, identities=len(ident),
        unassigned_chains=len(unassigned), unassigned_frames=sum(len(f) for f, _ in unassigned),
        visible_per_identity_pct=[round(100 * v) for v in sorted(vis_each, reverse=True)],
        mean_identified_per_frame=round(float(np.mean(counts)), 2),
        frames_with_all_N_pct=round(100 * float(np.mean([c == N for c in counts])), 1),
        frames_with_ge_N_minus_2_pct=round(100 * float(np.mean([c >= N - 2 for c in counts])), 1),
        raw_people_per_frame_mean=round(float(np.mean([len(o) for o in per_frame])), 2))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    render(meta, per_frame, rows, ident, unassigned, sfps)
    print("wrote", OUT / "ids.webm")


if __name__ == "__main__":
    have = len(open(OUT / "dets.jsonl").readlines()) if (OUT / "dets.jsonl").exists() else 0
    if (OUT / "meta.json").exists():                     # detections made by another script (e.g. clip_tiles_detect.py)
        m_ = json.load(open(OUT / "meta.json"))
        need = int(SECONDS * m_["fps"]) // m_["stride"]
    else:
        need = SECONDS * 15 - 2
    if have < 0.98 * need:
        detect()
    analyse()
