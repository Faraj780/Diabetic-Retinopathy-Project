#!/usr/bin/env python3
"""Train YOLOv8 on the prepared data (run scripts/prepare_data.py first).

    python scripts/train.py --epochs 1                        # quick check that training runs
    python scripts/train.py --size n --imgsz 1024 --epochs 50
    python scripts/train.py --resume runs/yolov8/<name>/weights/last.pt

Any other Ultralytics setting: --set lr0=0.001 --set mosaic=0
Each run is saved in runs/yolov8/<name>/. Only train and val are used here; the test split is
evaluated once, at the end (see README).

GPUs: only those in training.allowed_gpus (2 and 3) are used; the others are hidden from
PyTorch. By default both are used; --device 2 or --device 3 uses one. --batch is the total.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from drdet.config import load_config, resolve_path  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--size", default="n", choices=["n", "s", "m", "l", "x"], help="model size (n = smallest)")
    parser.add_argument("--imgsz", type=int, default=1024, help="training image size; small lesions vanish at 640")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16, help="total over all GPUs; lower it if a GPU runs out of memory")
    parser.add_argument("--device", help="GPU ids from training.allowed_gpus, e.g. 2 or 2,3 (default: all of them)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", help="run name (default built from the settings)")
    parser.add_argument("--resume", type=Path, metavar="LAST_PT", help="resume an interrupted run from its weights/last.pt")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra Ultralytics setting")
    return parser.parse_args(argv)


def restrict_gpus(device: str | None, allowed: list[int]) -> str:
    """Hide every GPU except the allowed ones and return the requested ones as torch ids.

    Sets CUDA_VISIBLE_DEVICES, so PyTorch (and Ultralytics' multi-GPU worker processes, which
    inherit it) can't see the other GPUs. Must run before torch is imported. The visible GPUs
    are renumbered from 0: with allowed [2, 3], GPU 2 becomes torch device 0 and GPU 3 device 1.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, allowed))
    requested = (device or ",".join(map(str, allowed))).split(",")
    if not all(gpu.strip().isdigit() and int(gpu) in allowed for gpu in requested) or len(set(requested)) != len(requested):
        raise SystemExit(f"--device {device}: only GPUs {allowed} may be used (training.allowed_gpus in the config)")
    return ",".join(str(allowed.index(int(gpu))) for gpu in requested)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    device = restrict_gpus(args.device, list(config["training"]["allowed_gpus"]))
    from ultralytics import YOLO  # imported after restrict_gpus so the other GPUs stay hidden

    if args.resume:
        model = YOLO(str(args.resume))
        model.train(resume=True)
    else:
        data = resolve_path(config["paths"]["output"]) / "yolo" / "dataset.yaml"
        if not data.exists():
            raise SystemExit(f"{data} not found. Run scripts/prepare_data.py first.")
        extra = {key.strip(): yaml.safe_load(value) for key, value in (pair.split("=", 1) for pair in args.set)}
        model = YOLO(f"yolov8{args.size}.pt")  # downloads the pretrained weights the first time
        model.train(
            data=str(data), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=device,
            workers=args.workers, seed=args.seed, deterministic=True, project=str(PROJECT_ROOT / "runs" / "yolov8"),
            name=args.name or f"yolov8{args.size}_img{args.imgsz}_e{args.epochs}", **extra,
        )

    # Ultralytics writes one row per epoch (losses, precision, recall, mAP50, mAP50-95) to
    # <run>/results.csv. Keep a copy in the results folder, named after the run.
    run_dir = Path(model.trainer.save_dir)
    results = resolve_path(config["paths"]["results"])
    results.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / "results.csv", results / f"{run_dir.name}_results.csv")
    print(f"Run saved in {run_dir}; results copied to {results / f'{run_dir.name}_results.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
