"""Export matched fundus images and lesion boxes to an Ultralytics YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from dr_data import BOX_LABELS, LABEL_TO_ID, AnnotationData, BuildReport, ImageRecord, build_image_records, load_annotations


TRAIN_VAL_RATIOS = {"train": 0.85, "val": 0.15}
TRAIN_VAL_NAMES = tuple(TRAIN_VAL_RATIOS)
OUTPUT_SPLIT_NAMES = ("train", "val", "test")
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

    rng = random.Random(seed)
    patient_labels = {
        patient: Counter(box.label for record in patient_records for box in record.boxes)
        for patient, patient_records in patient_images.items()
    }
    label_patient_counts = Counter(
        label for labels in patient_labels.values() for label in labels
    )
    total_label_boxes = Counter(
        box.label for record in records for box in record.boxes
    )
    target_images = {
        split: TRAIN_VAL_RATIOS[split] * len(records) for split in TRAIN_VAL_NAMES
    }
    target_boxes = {
        split: {
            label: TRAIN_VAL_RATIOS[split] * total_label_boxes[label]
            for label in total_label_boxes
        }
        for split in TRAIN_VAL_NAMES
    }
    patients = sorted(
        patient_images,
        key=lambda patient: (
            min(
                (label_patient_counts[label] for label in patient_labels[patient]),
                default=len(patient_images) + 1,
            ),
            -len(patient_labels[patient]),
            -sum(patient_labels[patient].values()),
            rng.random(),
        ),
    )
    assignments = {}
    split_image_counts = Counter()
    split_box_counts = {split: Counter() for split in TRAIN_VAL_NAMES}

    for patient_index, patient in enumerate(patients):
        patient_records = patient_images[patient]
        label_counts = patient_labels[patient]

        def assignment_cost(split):
            image_before = split_image_counts[split] - target_images[split]
            image_after = image_before + len(patient_records)
            image_cost = (image_after**2 - image_before**2) / max(target_images[split] ** 2, 1)
            class_cost = 0.0
            for label, count in label_counts.items():
                target = target_boxes[split][label]
                before = split_box_counts[split][label] - target
                after = before + count
                class_cost += (after**2 - before**2) / max(target**2, 1)
            return image_cost + class_cost / max(len(label_counts), 1)

        if len(patients) >= 2 and patient_index == len(patients) - 1 and not split_image_counts["val"]:
            split = "val"
        else:
            split = min(TRAIN_VAL_NAMES, key=assignment_cost)
        split_image_counts[split] += len(patient_records)
        split_box_counts[split].update(label_counts)
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
        invalid = set(saved.values()) - set(TRAIN_VAL_NAMES)
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


def matching_annotated_images(annotation_data: AnnotationData, image_root: Path):
    return {
        filename
        for filename, rows in annotation_data.rows_by_image.items()
        if any((row.get("annotation_type") or "").strip().lower() == "rectangle" for row in rows)
        and (image_root / filename).is_file()
    }


def combine_train_test_assignments(train_records, test_records, train_assignments):
    train_patients = {patient_id(record.filename) for record in train_records}
    test_patients = {patient_id(record.filename) for record in test_records}
    shared_patients = train_patients & test_patients
    if shared_patients:
        raise ValueError(
            "Patient IDs occur in both train and test folders: "
            + ", ".join(sorted(shared_patients)[:10])
        )

    assignments = dict(train_assignments)
    for record in test_records:
        if record.filename in assignments:
            raise ValueError(f"Image occurs in both train and test folders: {record.filename}")
        assignments[record.filename] = "test"
    return assignments


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
    class_counts_by_split = {split: Counter() for split in OUTPUT_SPLIT_NAMES}
    for split in OUTPUT_SPLIT_NAMES:
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


def print_audit(annotation_data: AnnotationData, train_report, test_report,
                missing_image_names, train_patient_count, test_patient_count,
                assignments, class_counts_by_split, exported_boxes):
    all_unique_boxes = sum(
        1 for rows in annotation_data.rows_by_image.values() for row in rows
        if (row.get("annotation_type") or "").strip().lower() == "rectangle"
    )
    unavailable_boxes = sum(
        1 for name in missing_image_names + train_report.corrupt_images + test_report.corrupt_images
        for row in annotation_data.rows_by_image[name]
        if (row.get("annotation_type") or "").strip().lower() == "rectangle"
    )
    dropped = (
        train_report.malformed_boxes + test_report.malformed_boxes
        + train_report.nonpositive_boxes + test_report.nonpositive_boxes
        + train_report.outside_boxes + test_report.outside_boxes
    )
    accounted = exported_boxes + dropped + unavailable_boxes
    print(f"CSV files: {len(annotation_data.csv_files)}; columns: {', '.join(sorted(annotation_data.columns))}")
    print(f"Raw rows: {annotation_data.raw_rows}; exact duplicate rows removed: {annotation_data.duplicate_rows}")
    print(f"Unique annotation types: {sorted(annotation_data.annotation_types)}")
    print(f"Unique labels: {sorted(annotation_data.label_counts)}")
    if annotation_data.scale_x and annotation_data.scale_y:
        print(f"image_scale_x range: {min(annotation_data.scale_x):.6f} to {max(annotation_data.scale_x):.6f}")
        print(f"image_scale_y range: {min(annotation_data.scale_y):.6f} to {max(annotation_data.scale_y):.6f}")
    print(f"Unique rectangle rows: {all_unique_boxes}; exported boxes: {exported_boxes}")
    print(f"Dropped malformed/unknown: {train_report.malformed_boxes + test_report.malformed_boxes}; nonpositive: {train_report.nonpositive_boxes + test_report.nonpositive_boxes}; outside: {train_report.outside_boxes + test_report.outside_boxes}")
    print(f"Missing image files: {len(missing_image_names)}; corrupt image files: {len(train_report.corrupt_images) + len(test_report.corrupt_images)}")
    print(f"Reconciliation (exported + dropped + missing/corrupt boxes): {accounted}/{all_unique_boxes}")
    print(f"Matched train images: {len(train_report.records)} across {train_patient_count} patients; matched test images: {len(test_report.records)} across {test_patient_count} patients")
    print(f"Export image counts: {dict(Counter(assignments.values()))}")
    for split in OUTPUT_SPLIT_NAMES:
        counts = class_counts_by_split[split]
        absent = sorted(set(BOX_LABELS) - set(counts))
        print(f"{split} boxes: {sum(counts.values())}; absent classes: {absent}")
    if accounted != all_unique_boxes:
        raise AssertionError("Box reconciliation failed")
    if train_patient_count < 2:
        print("Warning: fewer than two training patients are available; validation is not meaningful.")
    if missing_image_names:
        print("First annotated image names missing from both folders: " + ", ".join(missing_image_names[:10]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("BBoxAnnotations"))
    parser.add_argument("--train-images", type=Path, default=Path("train"))
    parser.add_argument("--test-images", type=Path, default=Path("test"))
    parser.add_argument("--output", type=Path, default=Path("outputs/yolo_dataset"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.train_images.is_dir():
        parser.error(f"Training image directory does not exist: {args.train_images}")
    annotation_data = load_annotations(args.annotations)
    train_names = matching_annotated_images(annotation_data, args.train_images)
    test_folder_available = args.test_images.is_dir()
    test_names = (
        matching_annotated_images(annotation_data, args.test_images)
        if test_folder_available
        else set()
    )
    overlapping_names = train_names & test_names
    if overlapping_names:
        parser.error("Image filenames occur in both folders: " + ", ".join(sorted(overlapping_names)[:10]))

    train_report = build_image_records(annotation_data, args.train_images, train_names)
    test_report = (
        build_image_records(annotation_data, args.test_images, test_names)
        if test_folder_available
        else BuildReport([], [], [])
    )
    if not train_report.records:
        parser.error(f"No annotated training images with usable boxes found under {args.train_images}")
    try:
        assignments = combine_train_test_assignments(
            train_report.records,
            test_report.records,
            {},
        )
    except ValueError as error:
        parser.error(str(error))

    train_assignments, train_patient_count, reused_splits = load_or_create_splits(
        train_report.records,
        args.output / "train_val_splits.csv",
        args.seed,
    )
    assignments = combine_train_test_assignments(
        train_report.records,
        test_report.records,
        train_assignments,
    )
    write_class_map(args.output)
    write_dataset_yaml(args.output)
    exported_boxes, class_counts_by_split = write_yolo_files(
        train_report.records + test_report.records,
        assignments,
        args.output,
    )
    all_box_names = {
        filename
        for filename, rows in annotation_data.rows_by_image.items()
        if any((row.get("annotation_type") or "").strip().lower() == "rectangle" for row in rows)
    }
    missing_image_names = sorted(all_box_names - train_names - test_names)
    print_audit(
        annotation_data,
        train_report,
        test_report,
        missing_image_names,
        train_patient_count,
        len({patient_id(record.filename) for record in test_report.records}),
        assignments,
        class_counts_by_split,
        exported_boxes,
    )
    print(f"Split file reused: {reused_splits}; YOLO dataset written to {args.output}")
    if not test_folder_available:
        print(f"Test image folder not present: {args.test_images}; test output is currently empty.")


if __name__ == "__main__":
    main()