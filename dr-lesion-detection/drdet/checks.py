"""Checks that the prepared YOLO data is what we think it is. Run via scripts/check_data.py."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .annotations import ImageRecord
from .export_yolo import parse_yolo_line
from .split import SPLITS


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    level: str = "error"  # "error" fails the run; "warning" is just reported


def _read_yolo(yolo_dir: Path):
    """{(split, stem): [label lines]} for every label file, and the image files per split."""
    labels, images = {}, {}
    for split in SPLITS:
        for label in sorted((yolo_dir / "labels" / split).glob("*.txt")):
            labels[(split, label.stem)] = [ln for ln in label.read_text().splitlines() if ln.strip()]
        images[split] = [p for p in sorted((yolo_dir / "images" / split).iterdir()) if not p.name.startswith(".")]
    return labels, images


def run_checks(output_dir: Path, class_names: list[str], records: list[ImageRecord] | None = None,
               tolerance_px: float = 1.0, open_images: bool = False) -> list[CheckResult]:
    """Run all checks. Pass records (freshly parsed from the CSVs) to also compare every
    label file against the original annotations."""
    output_dir = Path(output_dir)
    yolo_dir = output_dir / "yolo"
    results: list[CheckResult] = []

    def add(name, passed, detail, level="error"):
        results.append(CheckResult(name, bool(passed), detail, level))

    report_path = output_dir / "prepare_report.json"
    if not report_path.exists() or not (yolo_dir / "dataset.yaml").exists():
        add("prepared data exists", False, f"{report_path} or {yolo_dir / 'dataset.yaml'} not found. Run prepare_data.py first.")
        return results
    report = json.loads(report_path.read_text())

    # 1. Every rectangle row in the CSVs was either kept or dropped for a logged reason.
    dropped = sum(report["boxes_dropped"].values())
    add("CSV rows accounted for", report["rectangle_rows"] == report["boxes_kept"] + dropped,
        f"rectangle rows {report['rectangle_rows']} = kept {report['boxes_kept']} + dropped {dropped}")

    if report.get("class_order") and report["class_order"] != class_names:
        add("class order matches config", False, "the config's classes differ from the ones used to export; rerun prepare_data.py")

    # 2. Label files: count and format.
    labels, images = _read_yolo(yolo_dir)
    yolo_total = sum(len(lines) for lines in labels.values())
    add("YOLO label count", yolo_total == report["boxes_kept"], f"{yolo_total} label lines vs {report['boxes_kept']} kept boxes")

    bad_lines = []
    for (split, stem), lines in labels.items():
        for line in lines:
            parts = line.split()
            try:
                class_id = int(parts[0])
                cx, cy, w, h = (float(v) for v in parts[1:5])
                ok = (len(parts) == 5 and 0 <= class_id < len(class_names) and 0 < w <= 1 and 0 < h <= 1
                      and cx - w / 2 >= -1e-6 and cx + w / 2 <= 1 + 1e-6 and cy - h / 2 >= -1e-6 and cy + h / 2 <= 1 + 1e-6)
            except (ValueError, IndexError):
                ok = False
            if not ok:
                bad_lines.append(f"{split}/{stem}.txt: {line!r}")
    add("YOLO label format and ranges", not bad_lines,
        "all lines are 'class cx cy w h' with values inside the image" if not bad_lines
        else f"{len(bad_lines)} bad lines, e.g. {bad_lines[:3]}")

    # 3. Images and labels pair up, and images can be reached.
    broken, missing_labels = [], []
    for split, files in images.items():
        for image in files:
            if not image.exists():  # a symlink whose original image is gone
                broken.append(f"{split}/{image.name}")
            if (split, image.stem) not in labels:
                missing_labels.append(f"{split}/{image.name}")
    add("image files reachable", not broken,
        "every image (or symlink) points at a real file" if not broken
        else f"{len(broken)} broken, e.g. {broken[:3]}. Did you move or copy the folder? Rerun prepare_data.py where the images live.")
    orphan_labels = [f"{s}/{stem}.txt" for (s, stem) in labels if stem not in {p.stem for p in images[s]}]
    add("every label has an image and vice versa", not missing_labels and not orphan_labels,
        f"images without labels: {missing_labels[:3]}, labels without images: {orphan_labels[:3]}")

    # 4. No image or patient in more than one split, and the folders match splits.csv.
    where = defaultdict(set)
    for split, files in images.items():
        for image in files:
            where[image.stem].add(split)
    leaked = [stem for stem, splits in where.items() if len(splits) > 1]
    add("no image in two splits", not leaked, f"{len(leaked)} images appear in more than one split {leaked[:3]}")

    split_csv = output_dir / "splits.csv"
    if split_csv.exists():
        with open(split_csv, newline="", encoding="utf-8") as handle:
            split_rows = list(csv.DictReader(handle))
        patients = defaultdict(set)
        for row in split_rows:
            patients[row["patient_id"]].add(row["split"])
        split_patients = [p for p, s in patients.items() if len(s) > 1]
        add("no patient in two splits", not split_patients,
            f"{len(split_patients)} patients have eyes in different splits {split_patients[:3]}")
        exported = {(s, p.name) for s, files in images.items() for p in files}
        expected = {(row["split"], row["image_file"]) for row in split_rows}
        add("export matches splits.csv", exported == expected,
            "folders match splits.csv" if exported == expected else f"{len(exported ^ expected)} differences")
    else:
        add("splits.csv exists", False, f"{split_csv} not found")

    # 5. Round trip: label files, converted back to pixels, match the boxes parsed from the CSVs.
    if records is not None:
        split_of = {p.name: s for s, files in images.items() for p in files}
        mismatches = []
        for record in records:
            split = split_of.get(record.path.name)
            expected = sorted((class_names.index(b.label), b.x1, b.y1, b.x2, b.y2) for b in record.boxes)
            found = []
            for line in labels.get((split, record.stem), []):
                class_id, (x1, y1, x2, y2), _ = parse_yolo_line(line, record.width, record.height)
                found.append((class_id, x1, y1, x2, y2))
            found.sort()
            same = split is not None and len(expected) == len(found) and all(
                a[0] == b[0] and all(abs(p - q) <= tolerance_px for p, q in zip(a[1:], b[1:])) for a, b in zip(expected, found))
            if not same:
                mismatches.append(record.path.name)
        add("labels match the CSV annotations", not mismatches,
            f"every box matches within {tolerance_px} px" if not mismatches
            else f"{len(mismatches)} images differ, e.g. {mismatches[:3]}")

    # 6. Optional: open every image (slow on thousands of images).
    if open_images:
        unreadable = []
        for split, files in images.items():
            for image in files:
                try:
                    with Image.open(image) as opened:
                        opened.verify()
                except Exception:  # PIL raises several types for corrupt files
                    unreadable.append(f"{split}/{image.name}")
        add("images open", not unreadable, f"{len(unreadable)} unreadable, e.g. {unreadable[:3]}")

    # 7. Warnings: classes missing from val/test, and split sizes far from the targets.
    per_split = {s: Counter() for s in SPLITS}
    for (split, _), lines in labels.items():
        for line in lines:
            per_split[split][int(line.split()[0])] += 1
    absent = [f"{class_names[c]} in {s}" for c in range(len(class_names)) for s in ("val", "test")
              if sum(per_split[x][c] for x in SPLITS) > 0 and per_split[s][c] == 0]
    add("every class appears in val and test", not absent,
        "yes" if not absent else f"missing: {absent}. Their results will be meaningless.", level="warning")

    ratios = report.get("split", {}).get("ratios")
    total_images = sum(len(f) for f in images.values())
    if ratios and total_images:
        off = {s: round(len(images[s]) / total_images, 3) for s in SPLITS if abs(len(images[s]) / total_images - ratios[s]) > 0.03}
        add("split sizes near targets", not off, "within 3 percentage points" if not off else f"off target: {off}", level="warning")

    return results
