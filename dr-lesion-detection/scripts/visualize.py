#!/usr/bin/env python3
"""Draw boxes on a random sample of images.

    # Original CSV annotations (what the old plotter did)
    python scripts/visualize.py --source csv

    # Exported YOLO labels read back from disk: should look identical to the CSV view
    python scripts/visualize.py --source yolo --split train

    # Annotations (solid) plus a model's predictions (dashed). Make predictions with e.g.
    #   yolo detect predict model=runs/.../best.pt source=data/processed/yolo/images/val save_txt=True save_conf=True
    python scripts/visualize.py --source yolo --split val --predictions runs/detect/predict/labels

Use --zoom to crop each image to the area around its boxes, and --images NAME ... to pick images.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from drdet.annotations import build_records  # noqa: E402
from drdet.config import load_config, resolve_path  # noqa: E402
from drdet.export_yolo import read_label_file  # noqa: E402
from drdet.visualize import Panel, draw_panels  # noqa: E402


def pick(items, names, count, seed, key):
    if names:
        wanted = {Path(n).stem for n in names}
        return [item for item in items if Path(key(item)).stem in wanted]
    return random.Random(seed).sample(items, min(count, len(items)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source", choices=["csv", "yolo"], default="csv")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train", help="for --source yolo")
    parser.add_argument("--predictions", type=Path, help="folder of YOLO prediction .txt files to overlay")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--images-dir", type=Path, help="image folder, for --source csv")
    parser.add_argument("--output-dir", type=Path, help="processed data folder, for --source yolo")
    parser.add_argument("--images", nargs="*", help="specific image names to draw")
    parser.add_argument("--max-images", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zoom", action="store_true")
    parser.add_argument("--out", type=Path, help="output PNG (default: outputs/visualizations/<source>_<split>.png)")
    args = parser.parse_args(argv)
    if args.max_images < 1:
        parser.error("--max-images must be at least 1")

    config = load_config(args.config)
    class_names = list(config["classes"])
    panels = []

    if args.source == "csv":
        csv_cfg = config.get("csv", {})
        records, _ = build_records(
            resolve_path(args.annotations or config["paths"]["annotations"]),
            resolve_path(args.images_dir or config["paths"]["images"]),
            class_names,
            columns=csv_cfg.get("columns"),
            rectangle_types=csv_cfg.get("rectangle_types", ["rectangle"]),
            patient_regex=None,
        )
        for record in pick(records, args.images, args.max_images, args.seed, key=lambda r: r.path.name):
            panels.append(Panel(record.path.name, record.path, record.boxes))
    else:
        yolo_dir = resolve_path(args.output_dir or config["paths"].get("output", "data/processed")) / "yolo"
        image_dir = yolo_dir / "images" / args.split
        if not image_dir.is_dir():
            print(f"ERROR: {image_dir} not found. Run prepare_data.py first.", file=sys.stderr)
            return 1
        images = [p for p in sorted(image_dir.iterdir()) if not p.name.startswith(".")]
        for image in pick(images, args.images, args.max_images, args.seed, key=lambda p: p.name):
            with Image.open(image) as opened:
                width, height = opened.size
            boxes = [b for b, _ in read_label_file(yolo_dir / "labels" / args.split / f"{image.stem}.txt", width, height, class_names)]
            predictions = []
            if args.predictions:
                predictions = read_label_file(args.predictions / f"{image.stem}.txt", width, height, class_names)
            panels.append(Panel(image.name, image, boxes, predictions))

    if not panels:
        print("No images to draw. Check the paths, or the names passed to --images.", file=sys.stderr)
        return 1
    suffix = "csv" if args.source == "csv" else f"yolo_{args.split}" + ("_pred" if args.predictions else "")
    out = args.out or resolve_path(f"outputs/visualizations/{suffix}.png")
    draw_panels(panels, out, zoom=args.zoom)
    print(f"Saved {len(panels)} images to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
