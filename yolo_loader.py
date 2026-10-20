"""Export matched fundus images and lesion boxes to an Ultralytics YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from dr_data import BOX_LABELS, LABEL_TO_ID, AnnotationData, ImageRecord, build_image_records, load_annotations


SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_NAMES = tuple(SPLIT_RATIOS)
PATIENT_PATTERN = re.compile(r"^(?P<patient>.+)_(?P<eye>left|right)$", re.IGNORECASE)


def patient_id(filename: str) -> str:
    match = PATIENT_PATTERN.match(Path(filename).stem)
    return match.group("patient") if match else Path(filename).stem


def to_yolo(box, image_width: int, image_height: int, class_id: int) -> str:
    center_x = ((box.x_min + box.x_max) / 2) / image_width
    center_y = ((box.y_min + box.y_max) / 2) / image_height
    width = (box.x_max - box.x_min) / image_width
    height = (box.y_max - box.y_min) / image_height
    values = (center_x, center_y, width, height)
    if not all(0 <= value <= 1 for value in values) or width <= 0 or height <= 0:
        raise ValueError(f"Invalid normalized YOLO box values: {values}")
    return f"{class_id} {center_x:.10f} {center_y:.10f} {width:.10f} {height:.10f}"


def from_yolo(line: str, image_width: int, image_height: int):
    class_text, center_x, center_y, width, height = line.split()
    center_x = float(center_x) * image_width
    center_y = float(center_y) * image_height
    width = float(width) * image_width
    height = float(height) * image_height
    return int(class_text), (
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )


def create_split_assignments(records: list[ImageRecord], seed: int):
    patient_images: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        patient_images[patient_id(record.filename)].append(record)

    patients = sorted(patient_images)
    random.Random(seed).shuffle(patients)
    assignments = {}
    group_counts = Counter()
    minimum_nonempty = min(len(patients), len(SPLIT_NAMES))

    for index, patient in enumerate(patients):
        if index < minimum_nonempty:
            split = SPLIT_NAMES[index]
        else:
            split = max(
                SPLIT_NAMES,
                key=lambda name: SPLIT_RATIOS[name] * len(patients) - group_counts[name],
            )
        group_counts[split] += 1
        for record in patient_images[patient]:
            assignments[record.filename] = split
    return assignments, len(patients)


def load_or_create_splits(records: list[ImageRecord], split_path: Path, seed: int):
    current_names = {record.filename for record in records}
    if split_path.exists():
        with split_path.open("r", newline="", encoding="utf-8") as file:
            saved = {row["filename"]: row["split"] for row in csv.DictReader(file)}
        if set(saved) != current_names:
            raise RuntimeError(
                f"Existing split file {split_path} does not match available records. "
                "Preserve and review splits.csv instead of replacing it automatically."
            )
        invalid = set(saved.values()) - set(SPLIT_NAMES)
        if invalid:
            raise RuntimeError(f"Unknown split names in {split_path}: {sorted(invalid)}")
        patient_splits = defaultdict(set)
        for filename, split in saved.items():
            patient_splits[patient_id(filename)].add(split)
        conflicting_patients = [patient for patient, splits in patient_splits.items() if len(splits) > 1]
        if conflicting_patients:
            raise RuntimeError(
                f"Existing split file separates eyes from patient(s): {conflicting_patients[:10]}"
            )
        return saved, len({patient_id(name) for name in saved}), True

    assignments, patient_count = create_split_assignments(records, seed)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "patient_id", "split"])
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.filename):
            writer.writerow({
                "filename": record.filename,
                "patient_id": patient_id(record.filename),
                "split": assignments[record.filename],
            })
    return assignments, patient_count, False


def write_class_map(output_root: Path):
    mapping = {
        label: {"yolo_id": class_id, "torchvision_id": class_id + 1}
        for class_id, label in enumerate(BOX_LABELS)
    }
    (output_root / "class_map.json").write_text(
        json.dumps(mapping, indent=2) + "\n",
        encoding="utf-8",
    )


def write_dataset_yaml(output_root: Path):
    lines = [
        f"path: {json.dumps(str(output_root.resolve()))}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {class_id}: {json.dumps(label)}" for class_id, label in enumerate(BOX_LABELS))
    (output_root / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_image_link(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"Refusing to replace non-symlink image: {destination}")
    destination.symlink_to(source.resolve())


def write_yolo_files(records: list[ImageRecord], assignments: dict[str, str], output_root: Path):
    total_boxes = 0
    class_counts_by_split = {split: Counter() for split in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for record in records:
        split = assignments[record.filename]
        image_link = output_root / "images" / split / record.filename
        label_path = output_root / "labels" / split / f"{Path(record.filename).stem}.txt"
        create_image_link(record.path, image_link)
        label_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for box in record.boxes:
            class_id = LABEL_TO_ID[box.label]
            line = to_yolo(box, record.width, record.height, class_id)
            lines.append(line)
            class_counts_by_split[split][box.label] += 1
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        exported_lines = label_path.read_text(encoding="utf-8").splitlines()
        if len(exported_lines) != len(record.boxes):
            raise AssertionError(f"YOLO label count mismatch for {record.filename}")
        for box, line in zip(record.boxes, exported_lines):
            parsed_class, parsed_box = from_yolo(line, record.width, record.height)
            expected = (box.x_min, box.y_min, box.x_max, box.y_max)
            if parsed_class != LABEL_TO_ID[box.label] or max(
                abs(a - b) for a, b in zip(expected, parsed_box)
            ) >= 1:
                raise AssertionError(f"YOLO round-trip failed for {record.filename}: {line}")
        total_boxes += len(exported_lines)

    return total_boxes, class_counts_by_split


def print_audit(annotation_data: AnnotationData, report, patient_count, assignments,
                class_counts_by_split, exported_boxes):
    all_unique_boxes = sum(
        1 for rows in annotation_data.rows_by_image.values() for row in rows
        if (row.get("annotation_type") or "").strip().lower() == "rectangle"
    )
    unavailable_boxes = sum(
        1 for name in report.missing_images + report.corrupt_images
        for row in annotation_data.rows_by_image[name]
        if (row.get("annotation_type") or "").strip().lower() == "rectangle"
    )
    dropped = report.malformed_boxes + report.nonpositive_boxes + report.outside_boxes
    accounted = exported_boxes + dropped + unavailable_boxes
    print(f"CSV files: {len(annotation_data.csv_files)}; columns: {', '.join(sorted(annotation_data.columns))}")
    print(f"Raw rows: {annotation_data.raw_rows}; exact duplicate rows removed: {annotation_data.duplicate_rows}")
    print(f"Unique annotation types: {sorted(annotation_data.annotation_types)}")
    print(f"Unique labels: {sorted(annotation_data.label_counts)}")
    if annotation_data.scale_x and annotation_data.scale_y:
        print(f"image_scale_x range: {min(annotation_data.scale_x):.6f} to {max(annotation_data.scale_x):.6f}")
        print(f"image_scale_y range: {min(annotation_data.scale_y):.6f} to {max(annotation_data.scale_y):.6f}")
    print(f"Unique rectangle rows: {all_unique_boxes}; exported boxes: {exported_boxes}")
    print(f"Dropped malformed/unknown: {report.malformed_boxes}; nonpositive: {report.nonpositive_boxes}; outside: {report.outside_boxes}")
    print(f"Missing image files: {len(report.missing_images)}; corrupt image files: {len(report.corrupt_images)}")
    print(f"Reconciliation (exported + dropped + missing/corrupt boxes): {accounted}/{all_unique_boxes}")
    print(f"Available images: {len(report.records)}; patient groups: {patient_count}; split counts: {dict(Counter(assignments.values()))}")
    for split in SPLIT_NAMES:
        counts = class_counts_by_split[split]
        absent = sorted(set(BOX_LABELS) - set(counts))
        print(f"{split} boxes: {sum(counts.values())}; absent classes: {absent}")
    if accounted != all_unique_boxes:
        raise AssertionError("Box reconciliation failed")
    if patient_count < 3:
        print("Warning: fewer than three patient groups are available; validation/test are not meaningful.")
    if report.missing_images:
        print("First missing annotated image names: " + ", ".join(report.missing_images[:10]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("BBoxAnnotations"))
    parser.add_argument("--images", type=Path, default=Path("sample"))
    parser.add_argument("--output", type=Path, default=Path("outputs/yolo_dataset"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.images.is_dir():
        parser.error(f"Image directory does not exist: {args.images}")
    annotation_data = load_annotations(args.annotations)
    report = build_image_records(annotation_data, args.images)
    if not report.records:
        parser.error(f"No valid annotated images found under {args.images}")

    assignments, patient_count, reused_splits = load_or_create_splits(
        report.records,
        args.output / "splits.csv",
        args.seed,
    )
    write_class_map(args.output)
    write_dataset_yaml(args.output)
    exported_boxes, class_counts_by_split = write_yolo_files(
        report.records,
        assignments,
        args.output,
    )
    print_audit(
        annotation_data,
        report,
        patient_count,
        assignments,
        class_counts_by_split,
        exported_boxes,
    )
    print(f"Split file reused: {reused_splits}; YOLO dataset written to {args.output}")


if __name__ == "__main__":
    main()