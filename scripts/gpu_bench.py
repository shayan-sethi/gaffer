#!/usr/bin/env python3
"""GPU vs CPU speed for the same detection work. Run with the GPU environment: .local_data/venv_gpu/Scripts/python.exe scripts/gpu_bench.py <video> [start_frame] [n=20]
Times the person detector at the size we use (1920 wide band) and the ball detector, on cuda and on cpu, and prints seconds per frame."""
import sys, time
import cv2, torch
from ultralytics import YOLO
V = sys.argv[1]; F0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1750; N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
cap = cv2.VideoCapture(V); cap.set(1, F0); frames = []
for _ in range(N):
    ok, f = cap.read()
    if ok: frames.append(f[380:1080])
print(len(frames), "frames of", frames[0].shape)
for name, w, sz in (("person (yolov8n, 2560)", "yolov8n.pt", 2560), ("ball model (1920)", r".local_data\models\football-ball-detection.pt", 1920)):
    for dev in (["0", "cpu"] if torch.cuda.is_available() else ["cpu"]):
        m = YOLO(w)                                                              # a fresh model per device: ultralytics keeps the first device it was used on
        m(frames[0], imgsz=sz, device=dev, verbose=False)                        # warm-up
        if dev != "cpu": torch.cuda.synchronize()
        t0 = time.time()
        for f in frames: m(f, imgsz=sz, device=dev, verbose=False)
        if dev != "cpu": torch.cuda.synchronize()
        print(f"{name:26s} device {dev:>3s}: {(time.time() - t0) / len(frames):.3f} s/frame")
if torch.cuda.is_available(): print("peak GPU memory:", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB of 4")
