"""Small portability helpers shared by the pipeline scripts (Windows + NVIDIA, Mac + Apple GPU, or plain CPU).
  device.pick_device()            'cuda' | 'mps' | 'cpu'   (override with env DEVICE=cpu|cuda|mps)
  device.autocast(dev)            bfloat16 autocast on CUDA, a no-op elsewhere
  device.yolo_device()            what to pass to ultralytics: 0 | 'mps' | 'cpu'
  device.add_sam2_to_path()       import sam2 from the pip install if present, else from .local_data/ext/sam2_src/sam2-*
  device.video_writer(path,fps,size)   (writer, actual_path): VP80 .webm on Windows, mp4v .mp4 where VP80 is not available
  device.gpu_mem(dev)             peak GPU memory in GB (0 when not CUDA)
"""
import os, sys, contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pick_device():
    import torch
    want = os.environ.get("DEVICE", "").strip().lower()
    if want in ("cpu", "cuda", "mps"):
        return want
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")           # a few ops are missing on MPS: fall back to the CPU for those
        return "mps"
    return "cpu"


def autocast(dev):
    import torch
    return torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else contextlib.nullcontext()


def yolo_device():
    d = pick_device()
    return 0 if d == "cuda" else d


def gpu_mem(dev):
    import torch
    return torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0


def add_sam2_to_path():
    try:
        import sam2  # noqa: F401
        return
    except ImportError:
        pass
    src = sorted((ROOT / ".local_data" / "ext" / "sam2_src").glob("sam2-*"))
    if not src:
        sys.exit("SAM 2 is not installed: pip install git+https://github.com/facebookresearch/sam2.git   (needs Python >= 3.10), or see docs/SETUP_MAC.md")
    sys.path.insert(0, str(src[0]))


def sam2_checkpoint():
    p = ROOT / ".local_data" / "ext" / "sam2_ckpt" / "sam2.1_hiera_small.pt"
    if not p.exists():
        sys.exit(f"missing {p}: run  bash scripts/get_weights.sh")
    return str(p)


def video_writer(path, fps, size):
    """(VideoWriter, path actually used). Tries VP80/.webm first on Windows (plays in the desktop app), else mp4v/.mp4."""
    import cv2
    path = Path(path)
    tries = [(".webm", "VP80"), (".mp4", "mp4v")] if sys.platform.startswith("win") else [(".mp4", "mp4v"), (".webm", "VP80")]
    if path.suffix == ".mp4":
        tries = [(".mp4", "mp4v")]
    for ext, fourcc in tries:
        p = path.with_suffix(ext); w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w, p
    raise RuntimeError("could not open a video writer (tried webm/VP80 and mp4/mp4v)")
