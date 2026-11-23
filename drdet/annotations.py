"""Read the annotation CSVs into one record per image, with its boxes in pixels.

This is the one place where CSV rows become boxes.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

# File names look like "10_left.jpeg": patient 10, left eye.
PATIENT_PATTERN = re.compile(r"^(?P<patient>\d+)_(left|right)$")


@dataclass
class Box:
    """A lesion box in pixels: (x1, y1) top-left corner, (x2, y2) bottom-right."""

    label: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class ImageRecord:
    path: Path
    width: int
    height: int
    patient: str
    grade: int | None = None  # DR grade 0-4, used only to balance the split
    boxes: list[Box] = field(default_factory=list)


def read_rows(annotation_root: Path) -> dict[str, list[dict]]:
    """All CSV rows grouped by image name.

    Exact duplicate rows are skipped: the FullKaggleBatched and StageBatched folders
    contain many of the same rows (about 41,500).
    """
    rows_by_image = defaultdict(list)
    seen = set()
    for csv_path in sorted(Path(annotation_root).rglob("*.csv")):
        with open(csv_path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                key = tuple(sorted(row.items()))
                if key not in seen:
                    seen.add(key)
                    rows_by_image[row["image_name"]].append(row)
    return rows_by_image


def row_to_box(row: dict, label: str, width: int, height: int) -> Box | None:
    """Convert a rectangle row to a Box clipped to the image, or None if under 1 px is left.

    The annotation tool showed a resized image, so coordinates are divided by its scale.
    """
    scale_x, scale_y = float(row["image_scale_x"]), float(row["image_scale_y"])
    x1 = float(row["left_coordinate"]) / scale_x
    y1 = float(row["top_coordinate"]) / scale_y
    x2 = x1 + float(row["width"]) / scale_x
    y2 = y1 + float(row["height"]) / scale_y
    x1, y1, x2, y2 = max(x1, 0.0), max(y1, 0.0), min(x2, float(width)), min(y2, float(height))
    if x2 - x1 < 1 or y2 - y1 < 1:  # outside the image, or too thin
        return None
    return Box(label, x1, y1, x2, y2)


def build_records(annotation_root: Path, image_root: Path, class_names: list[str],
                  label_map: dict[str, str], grade_labels: dict[str, int]):
    """Parse every CSV. Returns (records of images with at least one box, Counter of dropped boxes)."""
    classes = set(class_names)
    records, dropped = [], Counter()

    for image_name, rows in sorted(read_rows(annotation_root).items()):
        rectangles = [row for row in rows if row["annotation_type"] == "rectangle"]
        if not rectangles:
            continue  # only whole-image labels, nothing to detect
        path = Path(image_root) / image_name
        if not path.is_file():
            dropped["image file missing"] += len(rectangles)
            continue
        with Image.open(path) as image:  # reads the header only, not the pixels
            width, height = image.size

        match = PATIENT_PATTERN.match(path.stem)
        grades = [grade_labels[row["label"]] for row in rows
                  if row["annotation_type"] == "whole_image" and row["label"] in grade_labels]
        record = ImageRecord(path, width, height,
                             patient=match["patient"] if match else path.stem,
                             grade=max(grades) if grades else None)  # graded twice: take the higher

        seen = set()
        for row in rectangles:
            label = label_map.get(row["label"], row["label"])
            if label not in classes:
                dropped[f"not a class: {label}"] += 1
                continue
            box = row_to_box(row, label, width, height)
            if box is None:
                dropped["outside the image"] += 1
                continue
            # The same lesion boxed twice (e.g. once as dot, once as generic haemorrhage): keep one.
            key = (label, round(box.x1 * 2), round(box.y1 * 2), round(box.x2 * 2), round(box.y2 * 2))
            if key in seen:
                dropped["duplicate box"] += 1
                continue
            seen.add(key)
            record.boxes.append(box)

        if record.boxes:
            records.append(record)
    return records, dropped
