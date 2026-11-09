"""Run a controlled YOLOv8 training run and evaluate its internal test split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics.data.augment import LetterBox
from ultralytics import YOLO

from yolo_loader import OUTPUT_SPLIT_NAMES, patient_id


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = REPO_ROOT / "outputs" / "yolo_dataset" / "dataset.yaml"
RUN_OUTPUT_DIR = REPO_ROOT / "outputs" / "yolo_runs"
RUN_NAME = "current"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_and_validate_dataset(dataset_yaml: Path):
    with dataset_yaml.open(encoding="utf-8") as file:
        dataset = yaml.safe_load(file)
    dataset_root = Path(dataset.get("path", dataset_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (dataset_yaml.parent / dataset_root).resolve()

    assignments_path = dataset_root / "patient_lesion_balanced_splits.csv"
    if not assignments_path.is_file():
        raise ValueError(f"Missing internal split manifest: {assignments_path}")
    with assignments_path.open(newline="", encoding="utf-8") as file:
        assignments = {row["filename"]: row["split"] for row in csv.DictReader(file)}
    if not assignments or set(assignments.values()) != set(OUTPUT_SPLIT_NAMES):
        raise ValueError("The split manifest must contain non-empty train, val, and test partitions.")

    patient_splits: dict[str, set[str]] = {}
    for filename, split in assignments.items():
        patient_splits.setdefault(patient_id(filename), set()).add(split)
    leaked_patients = [patient for patient, splits in patient_splits.items() if len(splits) > 1]
    if leaked_patients:
        raise ValueError(f"Patient leakage detected in split manifest: {leaked_patients[:10]}")

    for split in OUTPUT_SPLIT_NAMES:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        images = {
            path.name for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        } if image_dir.is_dir() else set()
        expected = {name for name, assigned_split in assignments.items() if assigned_split == split}
        if not images or images != expected:
            raise ValueError(
                f"{split} images do not match the split manifest: "
                f"{len(images)} exported, {len(expected)} assigned."
            )
        missing_labels = [Path(name).stem + ".txt" for name in images
                          if not (label_dir / (Path(name).stem + ".txt")).is_file()]
        if missing_labels:
            raise ValueError(f"Missing {split} labels, for example: {missing_labels[:5]}")

    return dataset, dataset_root, assignments


def metric_value(metrics, name: str):
    value = getattr(metrics.box, name, 0.0)
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def create_resized_box_sample(dataset_root: Path, dataset, image_size: int, output_path: Path):
    label_dir = dataset_root / "labels" / "val"
    image_dir = dataset_root / "images" / "val"
    candidates = []
    for label_path in sorted(label_dir.glob("*.txt")):
        lines = label_path.read_text(encoding="utf-8").splitlines()
        if 3 <= len(lines) <= 18:
            candidates.append((len({line.split()[0] for line in lines}), -abs(len(lines) - 8), label_path))
    if not candidates:
        raise ValueError(f"No validation image with a readable number of boxes exists in {label_dir}")
    label_path = max(candidates)[2]
    image_path = next(
        (image_dir / f"{label_path.stem}{extension}" for extension in IMAGE_EXTENSIONS
         if (image_dir / f"{label_path.stem}{extension}").is_file()),
        None,
    )
    if image_path is None:
        raise FileNotFoundError(f"No exported image matches {label_path.name}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read sample image: {image_path}")
    source_height, source_width = image_bgr.shape[:2]
    resized_bgr = LetterBox(new_shape=(image_size, image_size))(image=image_bgr)
    scale = min(image_size / source_height, image_size / source_width)
    resized_width = int(round(source_width * scale))
    resized_height = int(round(source_height * scale))
    horizontal_padding = (image_size - resized_width) / 2
    vertical_padding = (image_size - resized_height) / 2
    pad_x = int(round(horizontal_padding - 0.1))
    pad_y = int(round(vertical_padding - 0.1))
    ratio_x = ratio_y = scale

    image = Image.fromarray(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    class_names = dataset["names"]
    if isinstance(class_names, list):
        class_names = dict(enumerate(class_names))
    else:
        class_names = {int(class_id): name for class_id, name in class_names.items()}
    font = ImageFont.load_default()
    class_box_counts = {}
    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id_text, center_x, center_y, box_width, box_height = line.split()
        class_id = int(class_id_text)
        center_x = float(center_x) * source_width
        center_y = float(center_y) * source_height
        box_width = float(box_width) * source_width
        box_height = float(box_height) * source_height
        x_min = (center_x - box_width / 2) * ratio_x + pad_x
        y_min = (center_y - box_height / 2) * ratio_y + pad_y
        x_max = (center_x + box_width / 2) * ratio_x + pad_x
        y_max = (center_y + box_height / 2) * ratio_y + pad_y
        color = (255, 64, 32)
        draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=2)
        class_box_counts[class_id] = class_box_counts.get(class_id, 0) + 1
        text = str(class_id)
        text_box = draw.textbbox((0, 0), text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_top = max(0, y_min - text_height - 4)
        draw.rectangle((x_min, label_top, min(image_size, x_min + text_width + 6), label_top + text_height + 4), fill=color)
        draw.text((x_min + 3, label_top + 2), text, fill=(255, 255, 255), font=font)

    legend_width = 400
    annotated = Image.new("RGB", (image_size + legend_width, image_size), (248, 248, 246))
    annotated.paste(image, (0, 0))
    legend = ImageDraw.Draw(annotated)
    y = 16
    legend.text((image_size + 16, y), f"Resize QA: {image_size} x {image_size}", fill=(30, 30, 30), font=font)
    y += 28
    legend.text((image_size + 16, y), f"Source: {source_width} x {source_height}", fill=(60, 60, 60), font=font)
    y += 32
    for class_id, count in sorted(class_box_counts.items()):
        label = f"{class_id}: {class_names.get(class_id, class_id)} ({count})"
        wrapped = textwrap.wrap(label, width=48)
        for line in wrapped:
            legend.text((image_size + 16, y), line, fill=(40, 40, 40), font=font)
            y += 16
        y += 5

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    print(
        f"Resize/box QA sample: {output_path} "
        f"(source {source_width}x{source_height} -> {image_size}x{image_size}, "
        f"{len(label_path.read_text(encoding='utf-8').splitlines())} boxes)"
    )
    return output_path


def write_test_metrics(metrics, class_names, output_csv: Path, run_name: str):
    fields = ["run", "split", "class_id", "class_name", "precision", "recall", "mAP50", "mAP50_95"]
    rows = [{
        "run": run_name,
        "split": "test",
        "class_id": "all",
        "class_name": "all",
        "precision": metric_value(metrics, "mp"),
        "recall": metric_value(metrics, "mr"),
        "mAP50": metric_value(metrics, "map50"),
        "mAP50_95": metric_value(metrics, "map"),
    }]
    box_metrics = metrics.box
    precision = getattr(box_metrics, "p", [])
    recall = getattr(box_metrics, "r", [])
    ap50 = getattr(box_metrics, "ap50", [])
    ap = getattr(box_metrics, "ap", [])
    class_ids = getattr(box_metrics, "ap_class_index", range(len(ap)))
    for index, class_id in enumerate(class_ids):
        class_id = int(class_id)
        rows.append({
            "run": run_name,
            "split": "test",
            "class_id": class_id,
            "class_name": class_names.get(class_id, str(class_id)),
            "precision": float(precision[index]) if len(precision) > index else 0.0,
            "recall": float(recall[index]) if len(recall) > index else 0.0,
            "mAP50": float(ap50[index]) if len(ap50) > index else 0.0,
            "mAP50_95": float(ap[index]) if len(ap) > index else 0.0,
        })
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1 or args.workers < 0:
        raise SystemExit("epochs, batch, and imgsz must be positive; workers cannot be negative.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise SystemExit("Launch through run_on_gpus_2_3.sh so only the approved physical GPUs 2 and 3 are visible.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Launch training through run_on_gpus_2_3.sh.")
    requested_devices = [item.strip() for item in args.device.split(",") if item.strip()]
    if len(requested_devices) > torch.cuda.device_count():
        raise SystemExit(
            f"Requested {len(requested_devices)} CUDA devices but only "
            f"{torch.cuda.device_count()} are visible."
        )

    dataset, dataset_root, assignments = load_and_validate_dataset(args.data.resolve())
    create_resized_box_sample(
        dataset_root,
        dataset,
        args.imgsz,
        REPO_ROOT / "outputs" / "visualizations" / f"resized_box_sample_{args.imgsz}.jpg",
    )
    run_dir = RUN_OUTPUT_DIR.resolve()
    if run_dir.exists():
        for item in run_dir.iterdir():
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "data": str(args.data.resolve()),
        "dataset_root": str(dataset_root),
        "split_images": {split: sum(value == split for value in assignments.values())
                          for split in OUTPUT_SPLIT_NAMES},
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "patience": args.patience,
        "deterministic": True,
        "amp": True,
        "optimizer": "auto",
        "cache": False,
        "rect": False,
        "cos_lr": False,
        "multi_scale": False,
        "augment": False,
        "train_augmentations": {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "translate": 0.1,
            "scale": 0.5,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
        },
        "mosaic_close_epoch_count": 10,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (run_dir / "training_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        amp=True,
        optimizer="auto",
        cache=False,
        rect=False,
        cos_lr=False,
        multi_scale=False,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        val=True,
        plots=True,
        save=True,
        close_mosaic=min(10, args.epochs),
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
    )

    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise RuntimeError(f"Training completed without producing {best_weights}")
    best_model = YOLO(str(best_weights))
    test_metrics = best_model.val(
        data=str(args.data.resolve()),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        plots=False,
    )
    metrics_path = run_dir / "test_metrics.csv"
    write_test_metrics(test_metrics, dataset.get("names", {}), metrics_path, RUN_NAME)
    (run_dir / "training_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Best checkpoint: {best_weights}")
    print(f"Test metrics CSV: {metrics_path}")
    print(
        "Test summary: "
        f"precision={metric_value(test_metrics, 'mp'):.4f}, "
        f"recall={metric_value(test_metrics, 'mr'):.4f}, "
        f"mAP50={metric_value(test_metrics, 'map50'):.4f}, "
        f"mAP50-95={metric_value(test_metrics, 'map'):.4f}"
    )


if __name__ == "__main__":
    main()