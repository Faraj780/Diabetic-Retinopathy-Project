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
OUTPUT_SPLIT_NAMES = tuple(SPLIT_RATIOS)
SPLIT_NAMES = OUTPUT_SPLIT_NAMES
SPLIT_ALGORITHM_VERSION = "2"
PATIENT_PATTERN = re.compile(r"^(?P<patient>.+)_(?P<eye>left|right)$", re.IGNORECASE)


def default_output_root() -> Path:
    repo_root = Path(__file__).resolve().parent
    return repo_root / "outputs" / "yolo_dataset"


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
    patient_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    patient_label_image_counts: dict[str, Counter[str]] = defaultdict(Counter)
    patient_eye_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        patient = patient_id(record.filename)
        patient_images[patient].append(record)
        patient_label_counts[patient].update(box.label for box in record.boxes)
        patient_label_image_counts[patient].update({label: 1 for label in {box.label for box in record.boxes}})
        patient_eye_counts[patient]["left" if record.filename.lower().endswith("_left") else "right"] += 1

    total_label_boxes = Counter(
        box.label for record in records for box in record.boxes
    )
    total_label_images = Counter(
        label for record in records for label in {box.label for box in record.boxes}
    )
    total_label_patients = Counter(
        label for counts in patient_label_image_counts.values() for label in counts
    )
    total_eye_boxes = Counter()
    for patient, counts in patient_eye_counts.items():
        total_eye_boxes.update(counts)

    total_images = len(records)
    target_images = {
        split: SPLIT_RATIOS[split] * total_images for split in SPLIT_NAMES
    }
    target_label_boxes = {
        split: {
            label: SPLIT_RATIOS[split] * total_label_boxes[label]
            for label in total_label_boxes
        }
        for split in SPLIT_NAMES
    }
    target_label_images = {
        split: {
            label: SPLIT_RATIOS[split] * total_label_images[label]
            for label in total_label_images
        }
        for split in SPLIT_NAMES
    }
    target_eye_counts = {
        split: {
            eye: SPLIT_RATIOS[split] * total_eye_boxes.get(eye, 0)
            for eye in ("left", "right")
        }
        for split in SPLIT_NAMES
    }

    rng = random.Random(seed)
    patients = sorted(
        patient_images,
        key=lambda patient: (
            -sum(patient_label_counts[patient].values()),
            -sum(
                1 / max(total_label_images[label], 1)
                for label in patient_label_image_counts[patient]
            ),
            -len(patient_images[patient]),
            -patient_eye_counts[patient].get("left", 0),
            -patient_eye_counts[patient].get("right", 0),
            rng.random(),
        ),
    )

    assignments = {}
    split_image_counts = Counter()
    split_box_counts = {split: Counter() for split in SPLIT_NAMES}
    split_label_image_counts = {split: Counter() for split in SPLIT_NAMES}
    split_eye_counts = {split: Counter() for split in SPLIT_NAMES}

    for patient in patients:
        patient_records = patient_images[patient]
        label_counts = patient_label_counts[patient]
        label_image_counts = patient_label_image_counts[patient]
        eye_counts = patient_eye_counts[patient]

        def partition_cost(split, include_patient):
            image_count = split_image_counts[split]
            box_counts = split_box_counts[split]
            label_image_counts = split_label_image_counts[split]
            eye_counts = split_eye_counts[split]
            if include_patient:
                image_count += len(patient_records)
            image_error = ((image_count - target_images[split]) / max(total_images, 1)) ** 2

            eye_error = 0.0
            for eye in ("left", "right"):
                eye_count = eye_counts[eye]
                if include_patient:
                    eye_count += patient_eye_counts[patient][eye]
                target = target_eye_counts[split].get(eye, 0.0)
                eye_total = total_eye_boxes.get(eye, 0)
                eye_error += ((eye_count - target) / max(eye_total, 1)) ** 2

            image_class_error = 0.0
            box_class_error = 0.0
            for label in total_label_boxes:
                box_count = box_counts[label]
                image_count = label_image_counts[label]
                if include_patient:
                    box_count += label_counts.get(label, 0)
                    image_count += patient_label_image_counts[patient].get(label, 0)
                box_target = target_label_boxes[split].get(label, 0.0)
                image_target = target_label_images[split].get(label, 0.0)
                image_total = total_label_images[label]
                box_total = total_label_boxes[label]
                image_class_error += (
                    (image_count - image_target) / max(image_total, 3)
                ) ** 2
                box_class_error += (
                    (box_count - box_target) / max(box_total, 10)
                ) ** 2
            class_count = max(len(total_label_boxes), 1)
            return (
                10.0 * image_error
                + eye_error
                + 4.0 * image_class_error / class_count
                + 0.5 * box_class_error / class_count
            )

        def assignment_cost(split):
            coverage_penalty = sum(
                10.0
                for label in label_image_counts
                if total_label_patients[label] >= len(SPLIT_NAMES)
                and split_label_image_counts[split][label] > 0
                and any(
                    split_label_image_counts[other_split][label] == 0
                    for other_split in SPLIT_NAMES
                    if other_split != split
                )
            )
            return partition_cost(split, True) - partition_cost(split, False) + coverage_penalty

        split = min(SPLIT_NAMES, key=assignment_cost)
        split_image_counts[split] += len(patient_records)
        split_eye_counts[split].update({eye: eye_counts.get(eye, 0) for eye in ("left", "right")})
        split_box_counts[split].update(label_counts)
        split_label_image_counts[split].update(label_image_counts)
        for record in patient_records:
            assignments[record.filename] = split
    return assignments, len(patients)


def load_or_create_splits(records: list[ImageRecord], split_path: Path, seed: int):
    current_names = {record.filename for record in records}
    if split_path.exists():
        with split_path.open("r", newline="", encoding="utf-8") as file:
            saved_rows = list(csv.DictReader(file))
        saved = {row["filename"]: row["split"] for row in saved_rows}
        if set(saved) != current_names:
            raise RuntimeError(
                f"Existing split file {split_path} does not match available records. "
                "Preserve and review splits.csv instead of replacing it automatically."
            )
        current_version = all(
            row.get("algorithm_version") == SPLIT_ALGORITHM_VERSION
            for row in saved_rows
        )
        if current_version and set(saved.values()).issubset(set(SPLIT_NAMES)) and set(saved.values()) == set(SPLIT_NAMES):
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
        writer = csv.DictWriter(file, fieldnames=["filename", "patient_id", "split", "algorithm_version"])
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.filename):
            writer.writerow({
                "filename": record.filename,
                "patient_id": patient_id(record.filename),
                "split": assignments[record.filename],
                "algorithm_version": SPLIT_ALGORITHM_VERSION,
            })
    return assignments, patient_count, False


def matching_annotated_images(annotation_data: AnnotationData, image_root: Path):
    return {
        filename
        for filename, rows in annotation_data.rows_by_image.items()
        if any((row.get("annotation_type") or "").strip().lower() == "rectangle" for row in rows)
        and (image_root / filename).is_file()
    }


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
    class_image_counts_by_split = {split: Counter() for split in OUTPUT_SPLIT_NAMES}
    for split in OUTPUT_SPLIT_NAMES:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for record in records:
        split = assignments[record.filename]
        for other_split in OUTPUT_SPLIT_NAMES:
            if other_split == split:
                continue
            old_image = output_root / "images" / other_split / record.filename
            if old_image.is_symlink() and old_image.resolve() == record.path.resolve():
                old_image.unlink()
            old_label = output_root / "labels" / other_split / f"{Path(record.filename).stem}.txt"
            if old_label.is_file():
                old_label.unlink()
        image_link = output_root / "images" / split / record.filename
        label_path = output_root / "labels" / split / f"{Path(record.filename).stem}.txt"
        create_image_link(record.path, image_link)
        label_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        class_image_counts_by_split[split].update({
            box.label for box in record.boxes
        })
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

    return total_boxes, class_counts_by_split, class_image_counts_by_split


def write_split_lesion_counts(output_root: Path, class_counts_by_split, class_image_counts_by_split):
    csv_path = output_root / "lesion_counts_by_split.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "lesion_type", "images_with_lesion", "lesion_box_count"],
        )
        writer.writeheader()
        for split in OUTPUT_SPLIT_NAMES:
            for label in BOX_LABELS:
                writer.writerow({
                    "split": split,
                    "lesion_type": label,
                    "images_with_lesion": class_image_counts_by_split[split][label],
                    "lesion_box_count": class_counts_by_split[split][label],
                })
    return csv_path


def write_audit_csv(annotation_data: AnnotationData, train_report,
                   missing_image_names, train_patient_count, test_patient_count,
                   assignments, class_counts_by_split, class_image_counts_by_split, exported_boxes,
                   output_root: Path, split_file_reused: bool):
    all_unique_boxes = sum(
        1 for rows in annotation_data.rows_by_image.values() for row in rows
        if (row.get("annotation_type") or "").strip().lower() == "rectangle"
    )
    unavailable_boxes = sum(
        1 for name in missing_image_names + train_report.corrupt_images
        for row in annotation_data.rows_by_image[name]
        if (row.get("annotation_type") or "").strip().lower() == "rectangle"
    )
    dropped = (
        train_report.malformed_boxes
        + train_report.nonpositive_boxes
        + train_report.outside_boxes
    )
    accounted = exported_boxes + dropped + unavailable_boxes
    if accounted != all_unique_boxes:
        raise AssertionError(f"Box reconciliation failed: {accounted}/{all_unique_boxes}")
    rows = [
        {"metric": "csv_files", "split": "", "label": "", "value": len(annotation_data.csv_files)},
        {"metric": "raw_rows", "split": "", "label": "", "value": annotation_data.raw_rows},
        {"metric": "duplicate_rows_removed", "split": "", "label": "", "value": annotation_data.duplicate_rows},
        {"metric": "unique_rectangle_rows", "split": "", "label": "", "value": all_unique_boxes},
        {"metric": "exported_boxes", "split": "", "label": "", "value": exported_boxes},
        {"metric": "dropped_malformed_unknown", "split": "", "label": "", "value": train_report.malformed_boxes},
        {"metric": "dropped_nonpositive", "split": "", "label": "", "value": train_report.nonpositive_boxes},
        {"metric": "dropped_outside", "split": "", "label": "", "value": train_report.outside_boxes},
        {"metric": "missing_image_files", "split": "", "label": "", "value": len(missing_image_names)},
        {"metric": "corrupt_image_files", "split": "", "label": "", "value": len(train_report.corrupt_images)},
        {"metric": "reconciliation", "split": "", "label": "", "value": f"{accounted}/{all_unique_boxes}"},
        {"metric": "source_images", "split": "", "label": "", "value": len(train_report.records)},
        {"metric": "test_images", "split": "", "label": "", "value": sum(value == "test" for value in assignments.values())},
        {"metric": "source_patients", "split": "", "label": "", "value": train_patient_count},
        {"metric": "test_patients", "split": "", "label": "", "value": test_patient_count},
        {"metric": "split_file_reused", "split": "", "label": "", "value": str(split_file_reused)},
    ]
    for split in OUTPUT_SPLIT_NAMES:
        rows.append({"metric": "split_image_count", "split": split, "label": "", "value": sum(1 for value in assignments.values() if value == split)})
        rows.append({"metric": "split_box_count", "split": split, "label": "", "value": sum(class_counts_by_split[split].values())})
        for label, count in sorted(class_image_counts_by_split[split].items()):
            rows.append({"metric": "class_image_count", "split": split, "label": label, "value": count})
        for label, count in sorted(class_counts_by_split[split].items()):
            rows.append({"metric": "class_count", "split": split, "label": label, "value": count})
    csv_path = output_root / "dataset_audit_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "split", "label", "value"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def main():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=repo_root / "BBoxAnnotations")
    parser.add_argument("--train-images", type=Path, default=Path("/mnt/hdd2/faraj/train"))
    parser.add_argument("--output", type=Path, default=default_output_root())
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.train_images.is_dir():
        parser.error(f"Training image directory does not exist: {args.train_images}")
    annotation_data = load_annotations(args.annotations)
    source_names = matching_annotated_images(annotation_data, args.train_images)
    train_report = build_image_records(annotation_data, args.train_images, source_names)
    if not train_report.records:
        parser.error(f"No annotated training images with usable boxes found under {args.train_images}")
    assignments, train_patient_count, reused_splits = load_or_create_splits(
        train_report.records,
        args.output / "patient_lesion_balanced_splits.csv",
        args.seed,
    )
    write_class_map(args.output)
    write_dataset_yaml(args.output)
    exported_boxes, class_counts_by_split, class_image_counts_by_split = write_yolo_files(
        train_report.records,
        assignments,
        args.output,
    )
    lesion_counts_csv = write_split_lesion_counts(
        args.output,
        class_counts_by_split,
        class_image_counts_by_split,
    )
    all_box_names = {
        filename
        for filename, rows in annotation_data.rows_by_image.items()
        if any((row.get("annotation_type") or "").strip().lower() == "rectangle" for row in rows)
    }
    missing_image_names = sorted(all_box_names - source_names)
    test_patient_count = len({
        patient_id(filename) for filename, split in assignments.items() if split == "test"
    })
    audit_csv = write_audit_csv(
        annotation_data,
        train_report,
        missing_image_names,
        train_patient_count,
        test_patient_count,
        assignments,
        class_counts_by_split,
        class_image_counts_by_split,
        exported_boxes,
        args.output,
        reused_splits,
    )
    split_image_counts = Counter(assignments.values())
    print(
        f"Exported {len(train_report.records)} images across {train_patient_count} patients: "
        f"{dict(split_image_counts)}; {exported_boxes} boxes validated."
    )
    print(f"Split file reused: {reused_splits}; YOLO dataset written to {args.output}")
    print(f"Audit summary CSV written to {audit_csv}")
    print(f"Lesion counts spreadsheet written to {lesion_counts_csv}")
    for label in BOX_LABELS:
        patient_count = len({
            patient_id(record.filename) for record in train_report.records
            if any(box.label == label for box in record.boxes)
        })
        present_splits = {
            assignments[record.filename]
            for record in train_report.records
            if any(box.label == label for box in record.boxes)
        }
        if patient_count < 3 or len(present_splits) < 3:
            print(f"Balance limitation: {label} occurs in {patient_count} patient groups across {len(present_splits)}/3 splits.")


if __name__ == "__main__":
    main()