"""Tests for the data pipeline. Run with:  python -m pytest -q"""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from conftest import CLASSES, PROJECT_ROOT, make_fake_dataset
from drdet.annotations import DEFAULT_COLUMNS, ImageRecord, Box, build_records, parse_patient, row_to_box
from drdet.checks import run_checks
from drdet.export_yolo import make_subset_dataset, parse_yolo_line, to_yolo_line
from drdet.pipeline import SplitMismatchError, prepare_dataset
from drdet.split import SPLITS, assign_splits, load_split_file
from drdet.stats import compute_stats, write_stats
from drdet.visualize import Panel, draw_panels


def prepare(fake, config, **kwargs):
    return prepare_dataset(config, annotations=fake["annotations"], images=fake["images"], grades_csv=fake["grades"],
                           output=fake["output"], log=lambda *_: None, **kwargs)


def fresh_records(fake, config):
    """Parse the CSVs again, as check_data.py does, to compare against the exported labels."""
    records, _ = build_records(fake["annotations"], fake["images"], CLASSES, patient_regex=config["split"]["patient_regex"])
    return records


# --- parsing ---------------------------------------------------------------

def test_row_to_box_divides_by_scale_and_clips():
    row = {"left_coordinate": 20, "top_coordinate": 40, "width": 10, "height": 8,
           "image_scale_x": 2, "image_scale_y": 2, "label": "x"}
    box, note = row_to_box(row, DEFAULT_COLUMNS, 400, 300)
    assert (box.x1, box.y1, box.x2, box.y2) == (10, 20, 15, 24) and note == "ok"

    row.update(left_coordinate=-4, image_scale_x=1, image_scale_y=1)  # hangs 4 px off the left edge
    box, note = row_to_box(row, DEFAULT_COLUMNS, 400, 300)
    assert (box.x1, box.x2, note) == (0, 6, "clipped")


def test_row_to_box_drop_reasons():
    base = {"left_coordinate": 10, "top_coordinate": 10, "width": 10, "height": 10, "label": "x"}
    assert row_to_box({**base, "width": "abc"}, DEFAULT_COLUMNS, 100, 100) == (None, "unparseable_coordinates")
    assert row_to_box({**base, "width": "nan"}, DEFAULT_COLUMNS, 100, 100) == (None, "unparseable_coordinates")
    assert row_to_box({**base, "left_coordinate": 200}, DEFAULT_COLUMNS, 100, 100) == (None, "outside_image")
    assert row_to_box({**base, "width": 0.3}, DEFAULT_COLUMNS, 100, 100) == (None, "too_small")
    box, note = row_to_box({**base, "left_coordinate": 95}, DEFAULT_COLUMNS, 100, 100)
    assert note == "clipped" and box.x2 == 100 and box.x1 == 95


def test_parse_patient():
    pattern = r"^(?P<patient>\d+)_(?P<eye>left|right)$"
    assert parse_patient("10_left", pattern) == ("10", "left", True)
    assert parse_patient("10_RIGHT", pattern) == ("10", "right", True)
    assert parse_patient("scan_7", pattern) == ("scan_7", "unknown", False)


def test_build_records_accounts_for_every_row(fake, config):
    records, report = build_records(fake["annotations"], fake["images"], CLASSES, patient_regex=config["split"]["patient_regex"])
    assert report["duplicate_csv_rows"] == 1
    assert report["boxes_kept"] == fake["good_boxes"]
    assert report["rectangle_rows"] == report["boxes_kept"] + sum(report["boxes_dropped"].values())
    assert report["boxes_dropped"] == {"duplicate_box": 1, "unknown_label": 1, "too_small": 1,
                                       "outside_image": 1, "image_file_missing": 1}
    assert report["boxes_adjusted"] == {"clipped": 1, "flipped": 1}
    assert report["annotation_types"]["polygon"] == 1
    assert report["images_missing_file"] == 1
    for record in records:
        for box in record.boxes:
            assert 0 <= box.x1 < box.x2 <= record.width and 0 <= box.y1 < box.y2 <= record.height


# --- splitting --------------------------------------------------------------

def test_split_keeps_patients_together_and_is_reproducible(fake, config):
    records, _ = build_records(fake["annotations"], fake["images"], CLASSES, patient_regex=config["split"]["patient_regex"])
    ratios = config["split"]["ratios"]
    first, _ = assign_splits(records, CLASSES, ratios, seed=3)
    second, _ = assign_splits(records, CLASSES, ratios, seed=3)
    assert first == second
    by_patient = defaultdict(set)
    for record in records:
        by_patient[record.patient_id].add(first[record.path.name])
    assert all(len(splits) == 1 for splits in by_patient.values())


def test_split_balances_classes_on_a_bigger_dataset():
    rng = np.random.default_rng(0)
    records = []
    for patient in range(600):
        for eye in ("left", "right"):
            boxes = []
            for _ in range(rng.integers(1, 12)):
                label = CLASSES[rng.choice(len(CLASSES), p=[0.5, 0.15, 0.1, 0.12, 0.08, 0.03, 0.02])]
                boxes.append(Box(0, 0, 5, 5, label))
            records.append(ImageRecord(f"{patient}_{eye}.jpeg", Path(f"{patient}_{eye}.jpeg"), 100, 100, boxes,
                                       str(patient), eye, int(rng.integers(1, 5))))
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    assignment, _ = assign_splits(records, CLASSES, ratios, seed=0, restarts=10)
    for label in CLASSES:
        counts = {s: 0 for s in SPLITS}
        for record in records:
            counts[assignment[record.path.name]] += sum(b.label == label for b in record.boxes)
        total = sum(counts.values())
        for split in SPLITS:
            assert abs(counts[split] / total - ratios[split]) < 0.03, (label, counts)
    images = {s: sum(assignment[r.path.name] == s for r in records) / len(records) for s in SPLITS}
    assert all(abs(images[s] - ratios[s]) < 0.02 for s in SPLITS), images


# --- exports and checks ------------------------------------------------------

def test_yolo_line_round_trip():
    box = Box(12.25, 30.5, 48.75, 61.0, CLASSES[2])
    class_id, (x1, y1, x2, y2), conf = parse_yolo_line(to_yolo_line(box, 2, 400, 300), 400, 300)
    assert class_id == 2 and conf is None
    assert np.allclose([x1, y1, x2, y2], [box.x1, box.y1, box.x2, box.y2], atol=0.01)


def test_full_prepare_then_checks_pass(fake, config):
    report = prepare(fake, config)
    out = fake["output"]
    assert (out / "yolo" / "dataset.yaml").exists() and (out / "splits.csv").exists()
    results = run_checks(out, CLASSES, fresh_records(fake, config), open_images=True)
    assert "labels match the CSV annotations" in {r.name for r in results}
    failed = [r for r in results if not r.passed and r.level == "error"]
    assert not failed, failed
    assert report["boxes_kept"] == fake["good_boxes"]
    assert not (out / "coco").exists()  # YOLO only for now


def test_checks_catch_a_corrupted_label(fake, config):
    prepare(fake, config)
    label = next((fake["output"] / "yolo" / "labels" / "train").glob("*.txt"))
    label.write_text(label.read_text() + "0 0.5 0.5 1.7 0.1\n")
    results = run_checks(fake["output"], CLASSES, fresh_records(fake, config))
    failed = {r.name for r in results if not r.passed and r.level == "error"}
    assert {"YOLO label format and ranges", "YOLO label count", "labels match the CSV annotations"} <= failed


def test_checks_catch_a_shifted_box(fake, config):
    """A label that is well-formed but in the wrong place is only caught by the CSV comparison."""
    prepare(fake, config)
    label = next((fake["output"] / "yolo" / "labels" / "val").glob("*.txt"))
    first, *rest = label.read_text().splitlines()
    parts = first.split()
    parts[1] = f"{float(parts[1]) * 0.9:.6f}"  # move the first box 10% to the left
    label.write_text("\n".join([" ".join(parts), *rest]) + "\n")
    results = {r.name: r for r in run_checks(fake["output"], CLASSES, fresh_records(fake, config))}
    assert results["YOLO label format and ranges"].passed
    assert not results["labels match the CSV annotations"].passed


def test_checks_catch_a_broken_symlink(fake, config):
    prepare(fake, config)
    image = next((fake["output"] / "yolo" / "images" / "val").glob("*.jpeg"))
    image.resolve().unlink()
    failed = {r.name for r in run_checks(fake["output"], CLASSES) if not r.passed}
    assert "image files reachable" in failed


def test_saved_split_is_reused(fake, config):
    prepare(fake, config)
    saved = load_split_file(fake["output"] / "splits.csv")
    prepare(fake, config, seed=99)  # a different seed must not change a saved split
    assert load_split_file(fake["output"] / "splits.csv") == saved


def test_new_images_require_resplit(fake, config, tmp_path):
    prepare(fake, config)
    extra = make_fake_dataset(tmp_path / "more", patients=30)
    with pytest.raises(SplitMismatchError):
        prepare_dataset(config, annotations=extra["annotations"], images=extra["images"],
                        output=fake["output"], log=lambda *_: None)


def test_overfit_subset(fake, config, tmp_path):
    prepare(fake, config)
    yaml_path = make_subset_dataset(fake["output"] / "yolo", tmp_path / "overfit", 5, CLASSES)
    subset = tmp_path / "overfit"
    assert len(list((subset / "images" / "train").iterdir())) == 5
    assert len(list((subset / "labels" / "train").glob("*.txt"))) == 5
    assert "val: images/train" in yaml_path.read_text()


def test_refuses_to_overwrite_unrelated_folder(fake, config):
    (fake["output"] / "yolo").mkdir(parents=True)
    (fake["output"] / "yolo" / "important.txt").write_text("not ours")
    with pytest.raises(FileExistsError):
        prepare(fake, config)


# --- outputs ------------------------------------------------------------------

def test_visualize_and_stats_write_files(fake, config, tmp_path):
    prepare(fake, config)
    records, _ = build_records(fake["annotations"], fake["images"], CLASSES)
    panels = [Panel(r.path.name, r.path, r.boxes, [(r.boxes[0], 0.9)]) for r in records[:3]]
    assert draw_panels(panels, tmp_path / "vis.png", zoom=True).stat().st_size > 0
    stats = compute_stats(fake["output"] / "yolo", CLASSES)
    assert sum(row["total"] for row in stats["per_class"].values()) == fake["good_boxes"]
    written = write_stats(stats, CLASSES, config["split"]["ratios"], tmp_path / "stats")
    assert all(p.exists() for p in written)
    assert "imgsz 1024" in (tmp_path / "stats" / "stats.md").read_text()


def run_script(*args):
    return subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT, capture_output=True, text=True)


def test_train_dry_run_yolov8_only(fake, config):
    prepare(fake, config)
    data = fake["output"] / "yolo" / "dataset.yaml"
    result = run_script("scripts/train.py", "--data", str(data), "--dry-run", "--set", "mosaic=0")
    assert result.returncode == 0, result.stderr
    assert "yolov8n.pt" in result.stdout and "mosaic" in result.stdout
    assert "splits_sha256: None" not in result.stdout  # the run is tied to the saved split
    result = run_script("scripts/train.py", "--data", str(data), "--size", "s", "--overfit", "10", "--dry-run")
    assert "yolov8s.pt" in result.stdout and "_overfit10" in result.stdout
    # Other models aren't available yet.
    assert run_script("scripts/train.py", "--model", "yolov5", "--dry-run").returncode == 2
