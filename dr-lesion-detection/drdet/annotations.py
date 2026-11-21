"""Read the bounding-box annotation CSVs into clean, per-image records.

This is the one place where CSV rows become boxes. The plotter, the exporters and
the checks all use it, so a parsing fix only has to be made here.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".tif", ".tiff", ".bmp")

# Our name for each field -> column name in the CSV. Override in the config if yours differ.
DEFAULT_COLUMNS = {
    "image_name": "image_name",
    "annotation_type": "annotation_type",
    "label": "label",
    "left": "left_coordinate",
    "top": "top_coordinate",
    "width": "width",
    "height": "height",
    "scale_x": "image_scale_x",
    "scale_y": "image_scale_y",
}

DEFAULT_PATIENT_REGEX = r"^(?P<patient>\d+)_(?P<eye>left|right)$"


@dataclass
class Box:
    """A lesion box in absolute pixels: (x1, y1) is the top-left corner, (x2, y2) the bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class ImageRecord:
    """One image and all of its boxes."""

    image_name: str  # the name as written in the CSV
    path: Path  # the image file on disk
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    patient_id: str = ""
    eye: str = "unknown"  # "left", "right" or "unknown"
    grade: int | None = None  # Kaggle DR grade, if a grades file was given

    @property
    def stem(self) -> str:
        return self.path.stem


def read_annotation_rows(annotation_root: Path, image_column: str = "image_name"):
    """Read every CSV under annotation_root and group rows by image name.

    Exact duplicate rows (identical in every column) are dropped, as in the original
    plotter. Returns (rows_by_image, stats).
    """
    annotation_root = Path(annotation_root)
    csv_paths = [annotation_root] if annotation_root.is_file() else sorted(annotation_root.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No annotation CSV files found under {annotation_root}")

    rows_by_image: dict[str, list[dict]] = defaultdict(list)
    seen: dict[str, set] = defaultdict(set)
    stats = Counter(csv_files=len(csv_paths))

    for csv_path in csv_paths:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or image_column not in reader.fieldnames:
                stats["csv_files_skipped_no_image_column"] += 1
                continue
            for row in reader:
                stats["csv_rows"] += 1
                image_name = (row.get(image_column) or "").strip()
                if not image_name:
                    stats["rows_without_image_name"] += 1
                    continue
                signature = tuple(sorted((str(key), str(value)) for key, value in row.items()))
                if signature in seen[image_name]:
                    stats["duplicate_csv_rows"] += 1
                    continue
                seen[image_name].add(signature)
                rows_by_image[image_name].append(row)

    return dict(rows_by_image), stats


def _scale(value) -> float:
    """Annotation-tool scale factor; anything missing or invalid counts as 1."""
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return 1.0
    return scale if math.isfinite(scale) and scale > 0 else 1.0


def row_to_box(row: dict, columns: dict, image_width: int, image_height: int, min_size: float = 1.0):
    """Convert one CSV row to a Box in original-image pixels.

    Returns (box, note). If the box can't be used, box is None and note says why.
    If it was usable but changed, note says how ("clipped", "flipped").
    """
    try:
        left = float(row[columns["left"]])
        top = float(row[columns["top"]])
        width = float(row[columns["width"]])
        height = float(row[columns["height"]])
    except (KeyError, TypeError, ValueError):
        return None, "unparseable_coordinates"
    if not all(math.isfinite(v) for v in (left, top, width, height)):
        return None, "unparseable_coordinates"

    # The annotation tool showed a resized image; dividing by its scale gives
    # coordinates in the original image (same as the plotter).
    scale_x = _scale(row.get(columns["scale_x"]))
    scale_y = _scale(row.get(columns["scale_y"]))
    left, width = left / scale_x, width / scale_x
    top, height = top / scale_y, height / scale_y

    notes = []
    if width < 0 or height < 0:  # box drawn right-to-left or bottom-to-top
        if width < 0:
            left, width = left + width, -width
        if height < 0:
            top, height = top + height, -height
        notes.append("flipped")

    x1, y1, x2, y2 = left, top, left + width, top + height
    if x2 <= 0 or y2 <= 0 or x1 >= image_width or y1 >= image_height:
        return None, "outside_image"

    # Clip to the image. Unlike the old plotter, this trims the box rather than
    # shifting it, so a box hanging off the left edge keeps its right edge.
    cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
    cx2, cy2 = min(x2, float(image_width)), min(y2, float(image_height))
    if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
        notes.append("clipped")
    if cx2 - cx1 < min_size or cy2 - cy1 < min_size:
        return None, "too_small"

    label = (row.get(columns["label"]) or "").strip()
    return Box(cx1, cy1, cx2, cy2, label), "+".join(notes) or "ok"


def find_image(image_root: Path, image_name: str) -> Path | None:
    """Find the file for a CSV image name, trying common extensions if needed."""
    name = Path(image_name)
    candidates = [image_root / name, image_root / name.name]
    stem = name.stem if name.suffix.lower() in IMAGE_EXTENSIONS else name.name
    for ext in IMAGE_EXTENSIONS:
        candidates += [image_root / f"{stem}{ext}", image_root / f"{stem}{ext.upper()}"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def parse_patient(stem: str, pattern: str | None):
    """Get (patient_id, eye, matched) from a file stem like '10_left'."""
    match = re.match(pattern, stem, flags=re.IGNORECASE) if pattern else None
    if not match:
        return stem, "unknown", False
    groups = match.groupdict()
    return groups.get("patient") or stem, (groups.get("eye") or "unknown").lower(), True


def load_grades(path: Path) -> dict[str, int]:
    """Read Kaggle's trainLabels.csv (columns image, level) into {image stem: grade}."""
    grades = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = {name.lower(): name for name in reader.fieldnames or []}
        image_key = fields.get("image") or fields.get("image_name") or fields.get("id_code")
        grade_key = fields.get("level") or fields.get("diagnosis") or fields.get("grade")
        if not image_key or not grade_key:
            raise ValueError(f"{path}: expected columns like 'image' and 'level', found {reader.fieldnames}")
        for row in reader:
            name = (row.get(image_key) or "").strip()
            try:
                grade = int(float(row.get(grade_key)))
            except (TypeError, ValueError):
                continue
            stem = Path(name).stem if Path(name).suffix.lower() in IMAGE_EXTENSIONS else name
            grades[stem] = grade
    return grades


def build_records(
    annotation_root: Path,
    image_root: Path,
    class_names: list[str],
    *,
    columns: dict | None = None,
    rectangle_types=("rectangle",),
    patient_regex: str | None = DEFAULT_PATIENT_REGEX,
    grades: dict[str, int] | None = None,
    min_box_size: float = 1.0,
    keep_images_without_boxes: bool = False,
):
    """Parse all CSVs into ImageRecords and a report of what was kept and dropped.

    Every rectangle row ends up either kept or counted under a reason in
    report["boxes_dropped"], so rectangle_rows == boxes_kept + sum(boxes_dropped).
    """
    columns = {**DEFAULT_COLUMNS, **(columns or {})}
    rect_types = {t.strip().lower() for t in rectangle_types}
    known_labels = set(class_names)
    image_root = Path(image_root)

    rows_by_image, csv_stats = read_annotation_rows(annotation_root, columns["image_name"])

    annotation_types, dropped, adjusted = Counter(), Counter(), Counter()
    unknown_labels, image_counts = Counter(), Counter()
    rectangle_rows = 0
    missing_examples: list[str] = []
    records_by_path: dict[Path, ImageRecord] = {}
    seen_boxes: dict[Path, set] = {}

    for image_name in sorted(rows_by_image):
        rects = []
        for row in rows_by_image[image_name]:
            kind = (row.get(columns["annotation_type"]) or "").strip().lower()
            annotation_types[kind or "(blank)"] += 1
            if kind in rect_types:
                rects.append(row)
        rectangle_rows += len(rects)

        path = find_image(image_root, image_name)
        if path is None:
            image_counts["missing_file"] += 1
            dropped["image_file_missing"] += len(rects)
            if len(missing_examples) < 5:
                missing_examples.append(image_name)
            continue

        record = records_by_path.get(path)
        if record is None:
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except (OSError, ValueError):
                image_counts["unreadable"] += 1
                dropped["image_unreadable"] += len(rects)
                continue
            patient_id, eye, matched = parse_patient(path.stem, patient_regex)
            if not matched:
                image_counts["unmatched_patient_pattern"] += 1
            grade = grades.get(path.stem) if grades else None
            record = ImageRecord(image_name, path, width, height, [], patient_id, eye, grade)
            records_by_path[path] = record
            seen_boxes[path] = set()
        else:
            # Two CSV names (e.g. "10_left" and "10_left.jpeg") point at the same file.
            image_counts["merged_duplicate_names"] += 1

        for row in rects:
            label = (row.get(columns["label"]) or "").strip()
            if label not in known_labels:
                unknown_labels[label or "(blank)"] += 1
                dropped["unknown_label"] += 1
                continue
            box, note = row_to_box(row, columns, record.width, record.height, min_box_size)
            if box is None:
                dropped[note] += 1
                continue
            if note != "ok":
                adjusted[note] += 1
            # Same lesion boxed twice (e.g. in two CSV files): keep one copy.
            key = (label, round(box.x1 * 2), round(box.y1 * 2), round(box.x2 * 2), round(box.y2 * 2))
            if key in seen_boxes[path]:
                dropped["duplicate_box"] += 1
                continue
            seen_boxes[path].add(key)
            record.boxes.append(box)

    records = []
    for record in records_by_path.values():
        if record.boxes or keep_images_without_boxes:
            records.append(record)
        else:
            image_counts["without_boxes"] += 1
    records.sort(key=lambda r: r.path.name)

    per_class = Counter(box.label for record in records for box in record.boxes)
    eyes_by_patient: dict[str, set] = defaultdict(set)
    for record in records:
        eyes_by_patient[record.patient_id].add(record.eye)

    report = {
        "csv_files": csv_stats["csv_files"],
        "csv_files_skipped_no_image_column": csv_stats["csv_files_skipped_no_image_column"],
        "csv_rows": csv_stats["csv_rows"],
        "duplicate_csv_rows": csv_stats["duplicate_csv_rows"],
        "rows_without_image_name": csv_stats["rows_without_image_name"],
        "annotation_types": dict(annotation_types),
        "rectangle_rows": rectangle_rows,
        "images_in_csv": len(rows_by_image),
        "images_missing_file": image_counts["missing_file"],
        "missing_file_examples": missing_examples,
        "images_unreadable": image_counts["unreadable"],
        "images_merged_duplicate_names": image_counts["merged_duplicate_names"],
        "images_without_boxes": image_counts["without_boxes"],
        "images_kept": len(records),
        "images_unmatched_patient_pattern": image_counts["unmatched_patient_pattern"],
        "images_with_grade": sum(r.grade is not None for r in records),
        "patients": len(eyes_by_patient),
        "patients_with_both_eyes": sum({"left", "right"} <= eyes for eyes in eyes_by_patient.values()),
        "boxes_kept": sum(per_class.values()),
        "boxes_kept_per_class": {name: per_class.get(name, 0) for name in class_names},
        "boxes_dropped": dict(dropped),
        "boxes_adjusted": dict(adjusted),
        "unknown_labels": dict(unknown_labels),
    }
    return records, report
