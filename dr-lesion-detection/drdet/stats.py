"""Dataset statistics from the exported YOLO data: class counts, split balance and box sizes.

The box-size numbers help choose the training image size: YOLO resizes the longest image
side to imgsz, so a lesion shrinks by imgsz / longest_side. Boxes that end up only a few
pixels across are very hard to detect.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from .export_yolo import read_label_file  # noqa: E402
from .split import SPLITS  # noqa: E402

SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}  # validated colour-blind-safe set
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
IMGSZ_OPTIONS = (640, 1024, 1280)
TINY_PX = 8  # a box this small after resizing is close to invisible to the detector


def compute_stats(yolo_dir: Path, class_names: list[str]) -> dict:
    yolo_dir = Path(yolo_dir)
    counts = {name: {s: 0 for s in SPLITS} for name in class_names}
    shorter_side = {name: [] for name in class_names}  # in original pixels
    longest_image_side = {name: [] for name in class_names}
    boxes_per_image, image_sizes, images_per_split = [], {}, {}

    for split in SPLITS:
        images = [p for p in sorted((yolo_dir / "images" / split).iterdir()) if not p.name.startswith(".")]
        images_per_split[split] = len(images)
        for image in images:
            with Image.open(image) as opened:  # reads only the file header, not the pixels
                width, height = opened.size
            image_sizes[f"{width}x{height}"] = image_sizes.get(f"{width}x{height}", 0) + 1
            boxes = read_label_file(yolo_dir / "labels" / split / f"{image.stem}.txt", width, height, class_names)
            boxes_per_image.append(len(boxes))
            for box, _ in boxes:
                if box.label in counts:
                    counts[box.label][split] += 1
                    shorter_side[box.label].append(min(box.width, box.height))
                    longest_image_side[box.label].append(max(width, height))

    per_class = {}
    for name in class_names:
        sides = np.array(shorter_side[name])
        longest = np.array(longest_image_side[name])
        row = {"total": sum(counts[name].values()), **counts[name]}
        if len(sides):
            row["median_shorter_side_px"] = float(np.median(sides))
            row["p10_shorter_side_px"] = float(np.percentile(sides, 10))
            for imgsz in IMGSZ_OPTIONS:
                resized = sides * imgsz / longest
                row[f"pct_under_{TINY_PX}px_at_{imgsz}"] = float((resized < TINY_PX).mean() * 100)
        per_class[name] = row

    all_sides = np.concatenate([np.array(v) for v in shorter_side.values()]) if any(shorter_side.values()) else np.array([])
    return {
        "per_class": per_class,
        "images_per_split": images_per_split,
        "boxes_per_image": boxes_per_image,
        "image_sizes": dict(sorted(image_sizes.items(), key=lambda kv: -kv[1])),
        "all_shorter_sides": all_sides.tolist(),
    }


def _style(axis):
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def _short(name: str) -> str:
    return name.replace("Retinal ", "").replace(" (Retinal Lipid Deposit)", "").replace(" (Cotton wool spot)", " (CWS)")


def write_stats(stats: dict, class_names: list[str], ratios: dict, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    per_class = stats["per_class"]
    labels = [_short(n) for n in class_names]
    y = np.arange(len(class_names))[::-1]

    # 1. Total boxes per class (one series, so no legend; counts written on the bars).
    figure, axis = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    _style(axis)
    totals = [per_class[n]["total"] for n in class_names]
    axis.barh(y, totals, color=SPLIT_COLORS["train"], height=0.6)
    for yi, total in zip(y, totals):
        axis.text(total, yi, f" {total}", va="center", fontsize=8, color=INK)
    axis.set_yticks(y, labels)
    axis.set_xlabel("boxes", color=MUTED, fontsize=9)
    axis.set_title("Boxes per lesion type (all splits)", loc="left", color=INK, fontsize=11)
    figure.tight_layout()
    written.append(out_dir / "class_counts.png")
    figure.savefig(written[-1], dpi=150)
    plt.close(figure)

    # 2. How each class's boxes are spread over the splits (100% bars, target lines).
    figure, axis = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    _style(axis)
    left = np.zeros(len(class_names))
    for split in SPLITS:
        share = np.array([per_class[n][split] / per_class[n]["total"] * 100 if per_class[n]["total"] else 0 for n in class_names])
        axis.barh(y, share, left=left, color=SPLIT_COLORS[split], height=0.6, edgecolor=SURFACE, linewidth=1.5, label=split)
        left += share
    boundary = 0.0
    for split in SPLITS[:-1]:
        boundary += ratios[split] * 100
        axis.axvline(boundary, color=INK, linewidth=1, linestyle=":")
    axis.set_xlim(0, 100)
    axis.set_yticks(y, labels)
    axis.set_xlabel("% of that class's boxes (dotted lines = target split)", color=MUTED, fontsize=9)
    axis.set_title("Split balance per lesion type", loc="left", color=INK, fontsize=11)
    axis.legend(ncols=3, loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    figure.tight_layout()
    written.append(out_dir / "class_split_balance.png")
    figure.savefig(written[-1], dpi=150)
    plt.close(figure)

    # 3. Box sizes in original pixels (log scale: lesions span orders of magnitude).
    sides = np.array(stats["all_shorter_sides"])
    if len(sides):
        figure, axis = plt.subplots(figsize=(8, 3.5), facecolor=SURFACE)
        _style(axis)
        bins = np.logspace(np.log10(max(sides.min(), 0.5)), np.log10(sides.max() + 1), 40)
        axis.hist(sides, bins=bins, color=SPLIT_COLORS["train"], edgecolor=SURFACE, linewidth=0.8)
        axis.set_xscale("log")
        axis.set_xlabel("shorter side of box, original image pixels (log scale)", color=MUTED, fontsize=9)
        axis.set_ylabel("boxes", color=MUTED, fontsize=9)
        axis.set_title("Lesion box sizes", loc="left", color=INK, fontsize=11)
        figure.tight_layout()
        written.append(out_dir / "box_sizes.png")
        figure.savefig(written[-1], dpi=150)
        plt.close(figure)

    # 4. Boxes per image.
    per_image = np.array(stats["boxes_per_image"])
    if len(per_image):
        figure, axis = plt.subplots(figsize=(8, 3.5), facecolor=SURFACE)
        _style(axis)
        axis.hist(per_image, bins=min(50, max(5, int(per_image.max()) + 1)), color=SPLIT_COLORS["train"], edgecolor=SURFACE, linewidth=0.8)
        axis.set_xlabel("boxes in one image", color=MUTED, fontsize=9)
        axis.set_ylabel("images", color=MUTED, fontsize=9)
        axis.set_title("Boxes per image", loc="left", color=INK, fontsize=11)
        figure.tight_layout()
        written.append(out_dir / "boxes_per_image.png")
        figure.savefig(written[-1], dpi=150)
        plt.close(figure)

    written.append(out_dir / "stats.md")
    written[-1].write_text(format_stats_markdown(stats, class_names))
    return written


def format_stats_markdown(stats: dict, class_names: list[str]) -> str:
    per_class = stats["per_class"]
    lines = ["# Dataset statistics", "", "## Boxes per class and split", "",
             "| class | total | " + " | ".join(SPLITS) + " | median shorter side (px) | 10th percentile (px) |",
             "|---|---:|" + "---:|" * len(SPLITS) + "---:|---:|"]
    for name in class_names:
        row = per_class[name]
        median = f"{row['median_shorter_side_px']:.1f}" if "median_shorter_side_px" in row else "-"
        p10 = f"{row['p10_shorter_side_px']:.1f}" if "p10_shorter_side_px" in row else "-"
        lines.append(f"| {name} | {row['total']} | " + " | ".join(str(row[s]) for s in SPLITS) + f" | {median} | {p10} |")

    lines += ["", f"## Boxes smaller than {TINY_PX} px after resizing to the training image size", "",
              "| class | " + " | ".join(f"imgsz {s}" for s in IMGSZ_OPTIONS) + " |",
              "|---|" + "---:|" * len(IMGSZ_OPTIONS)]
    for name in class_names:
        row = per_class[name]
        cells = [f"{row[f'pct_under_{TINY_PX}px_at_{s}']:.0f}%" if f"pct_under_{TINY_PX}px_at_{s}" in row else "-" for s in IMGSZ_OPTIONS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    per_image = np.array(stats["boxes_per_image"]) if stats["boxes_per_image"] else np.array([0])
    lines += ["", "## Images", "",
              "Images per split: " + ", ".join(f"{s} {n}" for s, n in stats["images_per_split"].items()),
              f"Boxes per image: median {np.median(per_image):.0f}, max {per_image.max()}",
              "Most common image sizes: " + ", ".join(f"{k} ({v})" for k, v in list(stats["image_sizes"].items())[:5]), ""]
    return "\n".join(lines)
