#!/usr/bin/env python3
"""
Can Fable 5.2 find the ball? 50-frame test (no manual labelling by you; the result is checked by eye on a contact sheet).

  python scripts/ball_llm_test.py prep     # cut 50 crops from the 1080p Tolworth minute (no network)
  python scripts/ball_llm_test.py run      # send them to the API (needs ANTHROPIC_API_KEY in YOUR terminal)
  python scripts/ball_llm_test.py sheet    # draw the answers on the crops and build contact sheets (no network)

Each crop is the area around the players at full 1080p resolution (never down-scaled, so a 5-10 px ball stays that big).
"""
import sys, os, json, base64, re, time
from pathlib import Path
import numpy as np
import cv2

D = Path(".local_data/tolworth_1080")
OUT = Path(".local_data/ball_llm")
MODEL = "claude-fable-5-2"
N_FRAMES = 50
PROMPT = (
    "This is a crop from a fixed camera filming a 5-a-side football match on turf, at full resolution. "
    "Find the MATCH BALL: a small, roughly round, usually white or light-coloured object, often near a player's feet or in the air. "
    "Ignore boots, heads, white shirts, goal posts, pitch markings and bags. The ball may be only 5-12 pixels wide. "
    f"The image is {{w}} x {{h}} pixels. Reply with JSON only, no other text: "
    '{{"visible": true or false, "x": <ball centre x in pixels>, "y": <ball centre y in pixels>, "confidence": <0 to 1>, "note": "<one short sentence on where it is>"}}. '
    "Pixel coordinates have the origin at the top-left of this image. If you cannot clearly see the ball, set visible to false and do not guess."
)


def prep():
    meta = json.load(open(D / "meta.json"))
    rows = sorted((json.loads(l) for l in open(D / "dets.jsonl")), key=lambda r: r["frame"])
    pick = sorted(set(list(np.linspace(8, len(rows) - 8, N_FRAMES - 1).astype(int)) + [420]))
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(meta["video"])
    info = []
    for i in pick:
        r = rows[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, f = cap.read()
        P = [d for d in r["dets"] if d[4] >= 0.3] or r["dets"]
        x1 = max(0, int(min(d[0] for d in P)) - 80)
        x2 = min(f.shape[1], int(max(d[2] for d in P)) + 80)
        y1 = max(0, int(min(d[1] for d in P)) - 80)
        y2 = min(f.shape[0], int(max(d[3] for d in P)) + 80)
        if x2 - x1 > 1500:                                    # keep it under the API's 1568 px long edge, trimming evenly
            cut = (x2 - x1 - 1500) // 2
            x1, x2 = x1 + cut, x2 - cut
        crop = f[y1:y2, x1:x2]
        name = f"row{i:04d}.jpg"
        cv2.imwrite(str(OUT / "crops" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        info.append(dict(row=int(i), file=name, x0=x1, y0=y1, w=x2 - x1, h=y2 - y1))
    (OUT / "frames.json").write_text(json.dumps(info, indent=1))
    print(f"{len(info)} crops in {OUT / 'crops'}; widths {min(c['w'] for c in info)}-{max(c['w'] for c in info)} px")


def run():
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set in this terminal.")
    client = anthropic.Anthropic()
    info = json.load(open(OUT / "frames.json"))
    done = {}
    if (OUT / "answers.json").exists():
        done = json.load(open(OUT / "answers.json"))
    tin = tout = 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else len(info)      # e.g. "run 10" = only the next 10 frames
    sent = 0
    for c in info:
        if c["file"] in done:
            continue
        if sent >= limit:
            break
        sent += 1
        b64 = base64.standard_b64encode((OUT / "crops" / c["file"]).read_bytes()).decode()
        msg = client.messages.create(
            model=MODEL, max_tokens=8000,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": PROMPT.format(w=c["w"], h=c["h"])}]}],
            # server-side fallback for refusals (default for this model), and a moderate effort level
            extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
            extra_body={"fallbacks": "default", "output_config": {"effort": "medium"}},
        )
        tin += msg.usage.input_tokens
        tout += msg.usage.output_tokens
        text = "".join(b.text for b in msg.content if b.type == "text")
        m = re.search(r"\{.*\}", text, re.S)
        try:
            ans = json.loads(m.group(0)) if m else {"visible": False, "note": "unparseable: " + text[:80]}
        except Exception:
            ans = {"visible": False, "note": "unparseable: " + text[:80]}
        ans["stop_reason"] = msg.stop_reason
        ans["model_used"] = getattr(msg, "model", MODEL)
        done[c["file"]] = ans
        (OUT / "answers.json").write_text(json.dumps(done, indent=1))
        print(c["file"], ans.get("visible"), ans.get("x"), ans.get("y"), ans.get("confidence"), "|", str(ans.get("note"))[:70], flush=True)
    print(f"done. tokens in {tin}, out {tout}")


def sheet():
    info = {c["file"]: c for c in json.load(open(OUT / "frames.json"))}
    ans = json.load(open(OUT / "answers.json"))
    tiles = []
    for name, a in ans.items():
        c = info[name]
        im = cv2.imread(str(OUT / "crops" / name))
        if a.get("visible") and a.get("x") is not None:
            x, y = int(a["x"]), int(a["y"])
            r = 60
            x1, y1 = max(0, x - r), max(0, y - r)
            patch = im[y1:y + r, x1:x + r].copy()
            if patch.size == 0:
                patch = np.zeros((2 * r, 2 * r, 3), np.uint8)
            cv2.circle(patch, (x - x1, y - y1), 14, (0, 0, 255), 1)
            patch = cv2.resize(patch, (360, 360), interpolation=cv2.INTER_CUBIC)
            label = f"{name} SAYS ball at ({x},{y}) conf {a.get('confidence')}"
        else:
            patch = cv2.resize(im, (360, int(360 * im.shape[0] / im.shape[1])))
            patch = cv2.copyMakeBorder(patch, 0, max(0, 360 - patch.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:360]
            label = f"{name} says NOT VISIBLE"
        cv2.putText(patch, label, (4, 14), 0, 0.42, (0, 255, 255), 1)
        tiles.append(patch)
    cols = 4
    per = 12
    for s in range(0, len(tiles), per):
        chunk = tiles[s:s + per]
        while len(chunk) % cols:
            chunk.append(np.zeros_like(tiles[0]))
        rows_ = [np.hstack(chunk[i:i + cols]) for i in range(0, len(chunk), cols)]
        cv2.imwrite(str(OUT / f"sheet_{s // per + 1}.jpg"), np.vstack(rows_))
    n_vis = sum(1 for a in ans.values() if a.get("visible"))
    print(f"{len(ans)} answers, {n_vis} say visible. Sheets: {len(tiles) // per + (len(tiles) % per > 0)}")


if __name__ == "__main__":
    {"prep": prep, "run": run, "sheet": sheet}[sys.argv[1]]()
