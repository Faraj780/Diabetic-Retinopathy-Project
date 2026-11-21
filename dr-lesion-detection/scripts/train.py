#!/usr/bin/env python3
"""Train YOLOv8 on the prepared data.

    python scripts/train.py --dry-run                     # print settings, don't train
    python scripts/train.py --overfit 10 --epochs 100     # sanity check: memorise 10 images
    python scripts/train.py --size n --imgsz 1024 --epochs 50

Extra Ultralytics settings: --set lr0=0.001 --set mosaic=0

Each run is saved under runs/<model>/<name>/ with run_info.json, and one line is added to
runs/experiments.csv so you can compare runs. The test split is never used here: evaluate it
once at the end (see README).

Only YOLOv8 is supported for now. Other models get added to MODELS once YOLOv8 works.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from drdet.config import load_config, resolve_path  # noqa: E402

# Supported models and the pretrained weights each starts from ({size} = n, s, m, l or x).
# Later models (YOLOv5, YOLO26, Faster R-CNN, DETR) will be added here.
MODELS = {
    "yolov8": {"weights": "yolov8{size}.pt"},
}
RUNS_DIR = PROJECT_ROOT / "runs"
EXPERIMENT_LOG = RUNS_DIR / "experiments.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolov8", choices=sorted(MODELS))
    parser.add_argument("--size", default="n", choices=["n", "s", "m", "l", "x"], help="model size (n = smallest)")
    parser.add_argument("--weights", help="weights file to start from instead of the pretrained one")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path, help="dataset YAML (default: <output>/yolo/dataset.yaml)")
    parser.add_argument("--imgsz", type=int, default=1024, help="training image size; small lesions vanish at 640")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8, help="lower this if the GPU runs out of memory")
    parser.add_argument("--device", default="0", help="GPU id like 0 or 0,1, or cpu. Check nvidia-smi for a free GPU")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", help="run name (default built from the settings)")
    parser.add_argument("--overfit", type=int, metavar="N", help="train and validate on the same N images (sanity check)")
    parser.add_argument("--resume", type=Path, metavar="LAST_PT", help="resume an interrupted run from its weights/last.pt")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra Ultralytics training setting")
    parser.add_argument("--dry-run", action="store_true", help="print what would run, then stop")
    return parser.parse_args(argv)


def parse_overrides(pairs: list[str]) -> dict:
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(value)
    return overrides


def file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def build_settings(args, config) -> dict:
    output = resolve_path(config["paths"].get("output", "data/processed"))
    data = (args.data if args.data else output / "yolo" / "dataset.yaml").resolve()
    output = data.parent.parent  # dataset.yaml sits in <output>/yolo/, next to splits.csv
    weights = args.weights or MODELS[args.model]["weights"].format(size=args.size)
    name = args.name or f"{args.model}{args.size}_img{args.imgsz}_e{args.epochs}_s{args.seed}"
    if args.overfit:
        name += f"_overfit{args.overfit}"
    return {
        "model": args.model,
        "weights": weights,
        "data": str(data),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "name": name,
        "project": str(RUNS_DIR / args.model),
        "overfit": args.overfit,
        "overrides": parse_overrides(args.set),
        "splits_csv": str(output / "splits.csv"),
        "splits_sha256": file_sha256(output / "splits.csv"),  # ties the run to the exact split used
    }


def prepare_overfit_data(settings: dict, config: dict) -> None:
    from drdet.export_yolo import make_subset_dataset

    yolo_dir = Path(settings["data"]).parent
    out_dir = RUNS_DIR / "_overfit_data" / f"n{settings['overfit']}_s{settings['seed']}"
    settings["data"] = str(make_subset_dataset(yolo_dir, out_dir, settings["overfit"], list(config["classes"]), seed=settings["seed"]))


def versions() -> dict:
    info = {"python": platform.python_version()}
    for module in ("ultralytics", "torch"):
        try:
            info[module] = __import__(module).__version__
        except ImportError:
            info[module] = None
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def record_run(save_dir: Path | None, settings: dict, metrics: dict) -> None:
    info = {"finished": datetime.now().isoformat(timespec="seconds"), "settings": settings, "versions": versions(), "metrics": metrics}
    if save_dir and Path(save_dir).is_dir():
        Path(save_dir, "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    new_file = not EXPERIMENT_LOG.exists()
    with open(EXPERIMENT_LOG, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["finished", "model", "weights", "name", "imgsz", "epochs", "batch", "seed", "overfit",
                             "precision", "recall", "mAP50", "mAP50-95", "split_sha256", "save_dir"])
        writer.writerow([info["finished"], settings["model"], settings["weights"], settings["name"], settings["imgsz"],
                         settings["epochs"], settings["batch"], settings["seed"], settings["overfit"] or "",
                         metrics.get("metrics/precision(B)", ""), metrics.get("metrics/recall(B)", ""),
                         metrics.get("metrics/mAP50(B)", ""), metrics.get("metrics/mAP50-95(B)", ""),
                         (settings["splits_sha256"] or "")[:12], save_dir or ""])


def train(settings: dict, resume: Path | None) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics isn't installed. Run: pip install -r requirements.txt")

    if resume:
        model = YOLO(str(resume))
        results = model.train(resume=True)
    else:
        model = YOLO(settings["weights"])  # downloads the pretrained weights the first time
        results = model.train(
            data=settings["data"], imgsz=settings["imgsz"], epochs=settings["epochs"], batch=settings["batch"],
            device=settings["device"], workers=settings["workers"], seed=settings["seed"], deterministic=True,
            project=settings["project"], name=settings["name"], **settings["overrides"],
        )
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    metrics = {k: round(float(v), 5) for k, v in (getattr(results, "results_dict", None) or {}).items()}
    record_run(save_dir, settings, metrics)
    print(f"\nRun saved in {save_dir}")
    if metrics:
        print("Validation metrics: " + ", ".join(f"{k} {v}" for k, v in metrics.items()))


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    settings = build_settings(args, config)

    if not args.resume and not Path(settings["data"]).exists():
        raise SystemExit(f"{settings['data']} not found. Run scripts/prepare_data.py first.")
    if not settings["splits_sha256"]:
        print("WARNING: splits.csv not found, so this run can't be tied to a split.")

    print("Training settings:")
    for key, value in settings.items():
        print(f"  {key}: {value}")
    if args.dry_run:
        print("\nDry run: nothing was trained.")
        return 0

    if settings["overfit"]:
        prepare_overfit_data(settings, config)
    train(settings, args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
