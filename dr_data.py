"""Shared annotation parsing and image-coordinate conversion."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image


BOX_LABELS = (
    "Diabetic retinal venous beading",
    "Hard Retinal Exudate (Retinal Lipid Deposit)",
    "New Vessels Elsewhere (NVE)",
    "New Vessels at Disc (NVD)",
    "Other Intraretinal Microvascular Abnormalities (IRMA)",
    "Pre-retinal Haemorrhage",
    "Retinal Blot Haemorrhage",
    "Retinal Dot Haemorrhage",
    "Retinal Haemorrhages",
    "Retinal Microaneurysm",
    "Retinal Thickening",
    "Retinal veins beaded (venous beading)",
    "Soft Retinal Exudate (Cotton wool spot)",
    "Vitreous Haemorrhage",
)
LABEL_TO_ID = {label: class_id for class_id, label in enumerate(BOX_LABELS)}


@dataclass(frozen=True)
class Box:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str


@dataclass
class ImageRecord:
    filename: str
    path: Path
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)


@dataclass
class AnnotationData:
    rows_by_image: dict[str, list[dict[str, str]]]
    csv_files: list[Path]
    columns: set[str]
    raw_rows: int
    duplicate_rows: int
    annotation_types: dict[str, int]
    label_counts: dict[str, int]
    scale_x: list[float]
    scale_y: list[float]


@dataclass
class BuildReport:
    records: list[ImageRecord]
    missing_images: list[str]
    corrupt_images: list[str]
    malformed_boxes: int = 0
    nonpositive_boxes: int = 0
    outside_boxes: int = 0


def is_box(row: dict[str, str]) -> bool:
    return (row.get("annotation_type") or "").strip().lower() == "rectangle"


def load_annotations(annotation_root: Path) -> AnnotationData:
    csv_files = sorted(annotation_root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No annotation CSV files found under {annotation_root}")

    rows_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_by_image: dict[str, set[Tuple[Tuple[str, Optional[str]], ...]]] = defaultdict(set)
    columns = set()
    raw_rows = 0
    duplicate_rows = 0
    annotation_types: dict[str, int] = defaultdict(int)
    label_counts: dict[str, int] = defaultdict(int)
    scale_x = []
    scale_y = []

    for csv_path in csv_files:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            columns.update(reader.fieldnames or [])
            for row in reader:
                raw_rows += 1
                image_name = (row.get("image_name") or "").strip()
                if not image_name:
                    continue
                row["image_name"] = image_name
                signature = tuple(sorted(row.items()))
                if signature in seen_by_image[image_name]:
                    duplicate_rows += 1
                    continue
                seen_by_image[image_name].add(signature)
                rows_by_image[image_name].append(row)
                annotation_type = (row.get("annotation_type") or "").strip()
                annotation_types[annotation_type] += 1
                label = (row.get("label") or "").strip()
                if label:
                    label_counts[label] += 1
                for field_name, values in (("image_scale_x", scale_x), ("image_scale_y", scale_y)):
                    try:
                        value = float(row.get(field_name) or "")
                    except (TypeError, ValueError):
                        continue
                    values.append(value)

    return AnnotationData(
        rows_by_image=dict(rows_by_image),
        csv_files=csv_files,
        columns=columns,
        raw_rows=raw_rows,
        duplicate_rows=duplicate_rows,
        annotation_types=dict(annotation_types),
        label_counts=dict(label_counts),
        scale_x=scale_x,
        scale_y=scale_y,
    )


def convert_box(row: dict[str, str], image_width: int, image_height: int):
    """Convert a CSV box to clamped absolute xyxy pixels, returning a drop reason."""
    try:
        left = float(row["left_coordinate"])
        top = float(row["top_coordinate"])
        width = float(row["width"])
        height = float(row["height"])
        scale_x = float(row.get("image_scale_x") or 1)
        scale_y = float(row.get("image_scale_y") or 1)
        label = (row.get("label") or "").strip()
    except (KeyError, TypeError, ValueError):
        return None, "malformed"

    if not all(map(lambda value: value == value and abs(value) != float("inf"),
                   (left, top, width, height, scale_x, scale_y))):
        return None, "malformed"
    if scale_x <= 0 or scale_y <= 0 or not label:
        return None, "malformed"
    if width <= 0 or height <= 0:
        return None, "nonpositive"
    if label not in LABEL_TO_ID:
        return None, "unknown_label"

    left /= scale_x
    top /= scale_y
    width /= scale_x
    height /= scale_y
    right = left + width
    bottom = top + height
    if right <= 0 or bottom <= 0 or left >= image_width or top >= image_height:
        return None, "outside"

    x_min = min(max(left, 0.0), float(image_width))
    y_min = min(max(top, 0.0), float(image_height))
    x_max = min(max(right, 0.0), float(image_width))
    y_max = min(max(bottom, 0.0), float(image_height))
    if x_max <= x_min or y_max <= y_min:
        return None, "outside"
    return Box(x_min, y_min, x_max, y_max, label), None


def build_image_records(
    annotation_data: AnnotationData,
    image_root: Path,
    filenames: Optional[set[str]] = None,
) -> BuildReport:
    records = []
    missing_images = []
    corrupt_images = []
    report = BuildReport(records, missing_images, corrupt_images)

    for filename, rows in sorted(annotation_data.rows_by_image.items()):
        if filenames is not None and filename not in filenames:
            continue
        box_rows = [row for row in rows if is_box(row)]
        image_path = image_root / filename
        if not image_path.is_file():
            missing_images.append(filename)
            continue
        if not box_rows:
            continue
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
        except (OSError, ValueError):
            corrupt_images.append(filename)
            continue

        record = ImageRecord(filename, image_path, width, height)
        for row in box_rows:
            box, reason = convert_box(row, width, height)
            if box:
                record.boxes.append(box)
            elif reason == "malformed":
                report.malformed_boxes += 1
            elif reason == "nonpositive":
                report.nonpositive_boxes += 1
            elif reason == "outside":
                report.outside_boxes += 1
            elif reason == "unknown_label":
                report.malformed_boxes += 1
        if record.boxes:
            records.append(record)
    return report