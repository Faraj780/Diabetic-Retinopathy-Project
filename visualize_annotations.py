"""Summarize DR annotations and overlay lesion boxes on fundus images."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from PIL import Image


GRADE_ORDER = [
    "No Diabetic Retinopathy",
    "Mild Non-Proliferative Diabetic Retinopathy",
    "Moderate Non-Proliferative Diabetic Retinopathy",
    "Severe Non-Proliferative Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy",
]
GRADE_NUMBER = {grade: index for index, grade in enumerate(GRADE_ORDER)}
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

    raw_row_count = 0
    for csv_path in csv_paths:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                raw_row_count += 1
                image_name = (row.get("image_name") or "").strip()
                if image_name:
                    signature = tuple(sorted(row.items()))
                    if signature not in seen_by_image[image_name]:
                        seen_by_image[image_name].add(signature)
                        rows_by_image[image_name].append(row)
    return rows_by_image, csv_paths, raw_row_count


def is_box(row):
    return (row.get("annotation_type") or "").strip().lower() == "rectangle"


def make_summary_plot(rows_by_image, output_path: Path):
    grade_counts = Counter()
    lesion_counts = Counter()
    box_areas = []

    for rows in rows_by_image.values():
        for row in rows:
            label = (row.get("label") or "").strip()
            if (row.get("annotation_type") or "").strip().lower() == "whole_image":
                grade_counts[label] += 1
            elif is_box(row):
                lesion_counts[label] += 1
                try:
                    box_areas.append(float(row["height"]) * float(row["width"]))
                except (KeyError, TypeError, ValueError):
                    pass

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    grades = [grade for grade in GRADE_ORDER if grade_counts[grade]]
    axes[0].barh(grades or ["No grade rows"], [grade_counts[g] for g in grades] or [0], color="#416788")
    axes[0].set_title("Image-level DR grades")
    axes[0].set_xlabel("Annotated images")
    axes[0].invert_yaxis()

    labels = [label for label, _ in lesion_counts.most_common()]
    axes[1].barh(labels or ["No lesion boxes"], [lesion_counts[label] for label in labels] or [0], color="#d16666")
    axes[1].set_title("Lesion bounding boxes")
    axes[1].set_xlabel("Boxes")
    axes[1].invert_yaxis()

    if box_areas:
        axes[2].hist(box_areas, bins=30, color="#4c956c", edgecolor="white")
        axes[2].set_xscale("log")
        axes[2].set_xlabel("Annotated area (CSV coordinate units, log scale)")
        axes[2].set_ylabel("Boxes")
    else:
        axes[2].text(0.5, 0.5, "No box dimensions", ha="center", va="center")
        axes[2].set_xticks([])
        axes[2].set_yticks([])
    axes[2].set_title("Bounding-box size distribution")

    figure.suptitle("Diabetic retinopathy annotation overview", fontsize=15)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


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


def make_overlay_plot(rows_by_image, image_root: Path, output_path: Path, max_images: int):
    selected = []
    image_paths = {}
    for name, rows in rows_by_image.items():
        if len(selected) >= max_images:
            break
        if not any(is_box(row) for row in rows):
            continue
        image_path = image_root / name
        if image_path.is_file():
            selected.append((name, rows))
            image_paths[name] = image_path
    if not selected:
        return 0

    columns = min(2, len(selected))
    rows_count = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows_count, columns, figsize=(14, 6 * rows_count), squeeze=False)
    for axis, (image_name, annotations) in zip(axes.flat, selected):
        box_annotations = [row for row in annotations if is_box(row)]
        with Image.open(image_paths[image_name]) as source_image:
            image = source_image.convert("RGB")
            axis.imshow(image)
            for annotation in box_annotations:
                draw_box(axis, annotation, image.width, image.height)
        grade = next(
            ((row.get("label") or "").strip() for row in annotations
             if (row.get("annotation_type") or "").strip().lower() == "whole_image"),
            "Grade unavailable",
        )
        axis.set_title(f"{image_name} | {grade} | {len(box_annotations)} boxes", fontsize=9)
        class_counts = Counter((row.get("label") or "Lesion").strip() for row in box_annotations)
        legend_handles = [
            Line2D([0], [0], color=LABEL_COLORS.get(label, "#f2c14e"), lw=2,
                   label=f"{label} ({count})")
            for label, count in sorted(class_counts.items())
        ]
        axis.legend(handles=legend_handles, loc="lower left", fontsize=6,
                    framealpha=0.88, ncol=1)
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
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("BBoxAnnotations"),
        help="Folder containing the batched annotation CSV files.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/images"),
        help="Folder containing the original fundus images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/visualizations"),
        help="Where PNG visualizations will be written.",
    )
    parser.add_argument("--max-images", type=int, default=6, help="Maximum box overlays to plot.")
    args = parser.parse_args()

    rows_by_image, csv_paths, raw_row_count = load_annotations(args.annotations)
    all_rows = [row for rows in rows_by_image.values() for row in rows]
    box_count = sum(is_box(row) for row in all_rows)
    grade_count = sum(
        (row.get("annotation_type") or "").strip().lower() == "whole_image"
        for row in all_rows
    )

    summary_path = args.output_dir / "annotation_summary.png"
    make_summary_plot(rows_by_image, summary_path)
    print(f"Loaded {len(csv_paths)} CSV files and {raw_row_count} raw rows; "
          f"kept {len(all_rows)} unique rows across {len(rows_by_image)} image names "
          f"({box_count} lesion boxes, {grade_count} image-grade rows).")
    print(f"Saved annotation summary: {summary_path}")

    if not args.images.exists():
        print(f"Image directory not found: {args.images}. Box overlays need the fundus image files.")
        return

    overlay_path = args.output_dir / "fundus_box_overlays.png"
    overlay_count = make_overlay_plot(rows_by_image, args.images, overlay_path, args.max_images)
    if overlay_count:
        print(f"Saved {overlay_count} fundus overlays: {overlay_path}")
    else:
        print(f"No matching images with rectangle annotations found under {args.images}.")


if __name__ == "__main__":
    main()