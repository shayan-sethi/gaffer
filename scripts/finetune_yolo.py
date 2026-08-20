"""Fine-tune yolov8n on the converted ball+player dataset.

Usage:
    python scripts/finetune_yolo.py [--epochs 60] [--imgsz 640] [--device mps]
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/ball_player_ft/data.yaml")
    parser.add_argument("--base", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--project", default="runs/finetune")
    parser.add_argument("--name", default="ball_player_v1")
    parser.add_argument("--mosaic", type=float, default=1.0)
    args = parser.parse_args()

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=15,
        pretrained=True,
        mosaic=args.mosaic,
    )


if __name__ == "__main__":
    main()
