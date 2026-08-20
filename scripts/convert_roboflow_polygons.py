"""Convert the Roboflow segmentation-polygon labels into YOLO detection
bounding boxes, and lay out a dataset dir ready for `ultralytics` training.

Each source label line is: `class x1 y1 x2 y2 ... xn yn` (normalized polygon).
We take the polygon's min/max envelope and emit `class cx cy w h` (normalized),
the standard YOLO detection format.

Source images are symlinked (not copied) into the new tree to avoid
duplicating ~1500 images on disk.
"""

import shutil
from pathlib import Path

SRC = Path("roboflow data")
DST = Path("datasets/ball_player_ft")
SPLITS = ["train", "valid", "test"]


def polygon_to_bbox(coords):
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return cx, cy, w, h


def convert_label_file(src_path: Path, dst_path: Path):
    lines_out = []
    for line in src_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = parts[0]
        coords = [float(v) for v in parts[1:]]
        if len(coords) == 4:
            # already a bbox line, pass through
            cx, cy, w, h = coords
        else:
            cx, cy, w, h = polygon_to_bbox(coords)
        lines_out.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    dst_path.write_text("\n".join(lines_out) + "\n")


def main():
    if DST.exists():
        shutil.rmtree(DST)

    counts = {}
    for split in SPLITS:
        src_images = SRC / split / "images"
        src_labels = SRC / split / "labels"
        dst_images = DST / split / "images"
        dst_labels = DST / split / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        n = 0
        for img_path in sorted(src_images.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            label_path = src_labels / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            (dst_images / img_path.name).symlink_to(img_path.resolve())
            convert_label_file(label_path, dst_labels / label_path.name)
            n += 1
        counts[split] = n

    data_yaml = DST / "data.yaml"
    data_yaml.write_text(
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "\n"
        "nc: 2\n"
        "names: ['ball', 'player']\n"
    )

    print("Converted dataset written to", DST.resolve())
    for split, n in counts.items():
        print(f"  {split}: {n} images")


if __name__ == "__main__":
    main()
