"""Draw lesion annotations on fundus images."""

from __future__ import annotations

import argparse
from pathlib import Path

from dr_data import build_image_records, load_annotations
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


def draw_box(axis, box):
    axis.add_patch(
        Rectangle(
            (box.x_min, box.y_min),
            box.x_max - box.x_min,
            box.y_max - box.y_min,
            fill=False,
            edgecolor=LABEL_COLORS.get(box.label, "#f2c14e"),
            linewidth=1.4,
        )
    )


def draw_annotated_images(records, output_path: Path, max_images: int):
    selected = records[:max_images]

    if not selected:
        return 0

    columns = min(2, len(selected))
    row_count = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(row_count, columns, figsize=(14, 6 * row_count), squeeze=False)

    for axis, record in zip(axes.flat, selected):
        with Image.open(record.path) as source_image:
            image = source_image.convert("RGB")
            axis.imshow(image)
            for box in record.boxes:
                draw_box(axis, box)

        labels = sorted({box.label for box in record.boxes})
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
        axis.set_title(record.filename, fontsize=9)
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

    if not args.images.is_dir():
        parser.error(f"Image directory does not exist: {args.images}")

    annotation_data = load_annotations(args.annotations)
    report = build_image_records(annotation_data, args.images)
    image_count = draw_annotated_images(report.records, args.output, args.max_images)
    if image_count:
        print(f"Saved box overlays for {image_count} images: {args.output}")
    else:
        print(f"No annotated images found in {args.images}")


if __name__ == "__main__":
    main()