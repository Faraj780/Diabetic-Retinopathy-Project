"""Draw lesion annotations on fundus images."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image


LABEL_COLORS = {
    "Hard Retinal Exudate (Retinal Lipid Deposit)": "#f2c14e",
    "New Vessels Elsewhere (NVE)": "#56b4e9",
    "New Vessels at Disc (NVD)": "#009e73",
    "Retinal Blot Haemorrhage": "#e76f51",
    "Retinal Dot Haemorrhage": "#cc79a7",
    "Soft Retinal Exudate (Cotton wool spot)": "#f0e442",
    "Retinal Microaneurysm": "#d55e00",
}


def load_annotations(annotation_root: Path):
    rows_by_image = defaultdict(list)
    seen_by_image = defaultdict(set)
    csv_paths = sorted(annotation_root.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No annotation CSV files found under {annotation_root}")

    for csv_path in csv_paths:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                image_name = (row.get("image_name") or "").strip()
                signature = tuple(sorted(row.items()))
                if image_name and signature not in seen_by_image[image_name]:
                    seen_by_image[image_name].add(signature)
                    rows_by_image[image_name].append(row)
    return rows_by_image


def is_box(row):
    return (row.get("annotation_type") or "").strip().lower() == "rectangle"


def draw_box(axis, row, image_width, image_height):
    try:
        left = float(row["left_coordinate"])
        top = float(row["top_coordinate"])
        width = float(row["width"])
        height = float(row["height"])
        scale_x = float(row.get("image_scale_x") or 1)
        scale_y = float(row.get("image_scale_y") or 1)
    except (KeyError, TypeError, ValueError):
        return

    if scale_x <= 0:
        scale_x = 1
    if scale_y <= 0:
        scale_y = 1

    left /= scale_x
    width /= scale_x
    top /= scale_y
    height /= scale_y
    left = min(max(left, 0), image_width)
    top = min(max(top, 0), image_height)
    width = min(width, image_width - left)
    height = min(height, image_height - top)
    label = (row.get("label") or "Lesion").strip()

    axis.add_patch(
        Rectangle(
            (left, top),
            width,
            height,
            fill=False,
            edgecolor=LABEL_COLORS.get(label, "#f2c14e"),
            linewidth=1.4,
        )
    )


def draw_annotated_images(rows_by_image, image_root: Path, output_path: Path, max_images: int):
    selected = []
    for image_name, annotations in rows_by_image.items():
        if len(selected) >= max_images:
            break
        boxes = [row for row in annotations if is_box(row)]
        image_path = image_root / image_name
        if boxes and image_path.is_file():
            selected.append((image_name, image_path, boxes))

    if not selected:
        return 0

    columns = min(2, len(selected))
    row_count = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(row_count, columns, figsize=(14, 6 * row_count), squeeze=False)

    for axis, (image_name, image_path, boxes) in zip(axes.flat, selected):
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
            axis.imshow(image)
            for annotation in boxes:
                draw_box(axis, annotation, image.width, image.height)

        labels = sorted({(row.get("label") or "Lesion").strip() for row in boxes})
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=LABEL_COLORS.get(label, "#f2c14e"),
                lw=2,
                label=label,
            )
            for label in labels
        ]
        axis.set_title(image_name, fontsize=9)
        axis.legend(handles=legend_handles, loc="lower left", fontsize=6, framealpha=0.88)
        axis.axis("off")

    for axis in axes.flat[len(selected):]:
        axis.axis("off")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("BBoxAnnotations"))
    parser.add_argument("--images", type=Path, default=Path("train"))
    parser.add_argument("--output", type=Path, default=Path("outputs/visualizations/fundus_box_overlays.png"))
    parser.add_argument("--max-images", type=int, default=6)
    args = parser.parse_args()

    if args.max_images < 1:
        parser.error("--max-images must be at least 1")

    rows_by_image = load_annotations(args.annotations)
    if not args.images.is_dir():
        parser.error(f"Image directory does not exist: {args.images}")

    image_count = draw_annotated_images(
        rows_by_image,
        args.images,
        args.output,
        args.max_images,
    )
    if image_count:
        print(f"Saved box overlays for {image_count} images: {args.output}")
    else:
        print(f"No annotated images found in {args.images}")


if __name__ == "__main__":
    main()