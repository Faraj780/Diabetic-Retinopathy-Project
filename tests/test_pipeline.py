"""Tests for data preparation and the GPU restriction. Run with:  python -m pytest -q"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import CLASSES, PROJECT_ROOT, row, write_csv
from drdet.annotations import Box, ImageRecord, build_records, row_to_box
from drdet.pipeline import prepare_dataset
from drdet.split import SPLITS, assign_splits, load_split


def test_row_to_box_divides_by_scale_and_clips():
    box = row_to_box(row("a", "x", 10, 20, 30, 40, scale=0.25), "x", 400, 300)
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((10, 20, 40, 60))
    box = row_to_box(row("a", "x", 390, 290, 30, 30), "x", 400, 300)  # hangs off the corner
    assert (box.x2, box.y2) == (400, 300)
    assert row_to_box(row("a", "x", 500, 10, 10, 10), "x", 400, 300) is None  # outside the image


def test_build_records(tmp_path):
    for name in ("1_left", "1_right"):
        Image.new("RGB", (100, 100)).save(tmp_path / f"{name}.jpeg")
    dot = row("1_left.jpeg", "Retinal Dot Haemorrhage", 10, 10, 5, 5)
    write_csv(tmp_path / "ann" / "a.csv", [
        dot,
        row("1_left.jpeg", "Retinal Haemorrhages", 10, 10, 5, 5),     # same lesion, generic label: duplicate
        row("1_left.jpeg", "Retinal Thickening", 20, 20, 5, 5),       # not a class: dropped
        row("1_left.jpeg", "No Diabetic Retinopathy", kind="whole_image"),
        row("1_left.jpeg", "Moderate DR", kind="whole_image"),       # graded twice: higher wins
        row("1_right.jpeg", "Retinal Microaneurysm", 30, 30, 4, 4),
        row("9_left.jpeg", "Retinal Microaneurysm", 30, 30, 4, 4),    # image file missing
    ])
    write_csv(tmp_path / "ann" / "b.csv", [dot])                       # exact duplicate row in another file
    label_map = {"Retinal Dot Haemorrhage": "Retinal Haemorrhage", "Retinal Haemorrhages": "Retinal Haemorrhage"}
    records, dropped = build_records(tmp_path / "ann", tmp_path, ["Retinal Microaneurysm", "Retinal Haemorrhage"],
                                     label_map, {"No Diabetic Retinopathy": 0, "Moderate DR": 2})

    by_name = {r.path.name: r for r in records}
    assert set(by_name) == {"1_left.jpeg", "1_right.jpeg"}
    assert [b.label for b in by_name["1_left.jpeg"].boxes] == ["Retinal Haemorrhage"]
    assert by_name["1_left.jpeg"].grade == 2 and by_name["1_right.jpeg"].grade is None
    assert by_name["1_left.jpeg"].patient == by_name["1_right.jpeg"].patient == "1"
    assert dropped == {"duplicate box": 1, "not a class: Retinal Thickening": 1, "image file missing": 1}


def test_label_map_targets_are_classes(config):
    assert set(config["label_map"].values()) <= set(config["classes"])


def test_prepare_writes_matching_labels_and_keeps_patients_together(config):
    yaml_path = prepare_dataset(config, log=lambda *_: None)
    records, _ = build_records(Path(config["paths"]["annotations"]), Path(config["paths"]["images"]), CLASSES,
                               config["label_map"], config["grade_labels"])
    split_of = load_split(yaml_path.parent.parent / "splits.csv")

    patient_splits = defaultdict(set)
    for record in records:
        split = split_of[record.path.name]
        patient_splits[record.patient].add(split)
        assert (yaml_path.parent / "images" / split / record.path.name).resolve() == record.path.resolve()
        # Every label line converts back to its box (within 1 px).
        lines = (yaml_path.parent / "labels" / split / f"{record.path.stem}.txt").read_text().split("\n")[:-1]
        assert len(lines) == len(record.boxes)
        for line, box in zip(lines, record.boxes):
            class_id, cx, cy, w, h = (float(v) for v in line.split())
            assert CLASSES[int(class_id)] == box.label
            assert abs((cx - w / 2) * record.width - box.x1) < 1 and abs((cy + h / 2) * record.height - box.y2) < 1
    assert all(len(splits) == 1 for splits in patient_splits.values())

    # The split summary counts every box of every class once.
    with open(Path(config["paths"]["results"]) / "split_summary.csv", newline="") as handle:
        summary = {row["count"]: row for row in csv.DictReader(handle)}
    assert int(summary["images"]["total"]) == len(records)
    for name in CLASSES:
        expected = sum(b.label == name for r in records for b in r.boxes)
        row = summary[f"boxes: {name}"]
        assert int(row["total"]) == expected == sum(int(row[s]) for s in SPLITS)


def test_saved_split_is_reused_and_new_images_need_resplit(config):
    prepare_dataset(config, log=lambda *_: None)
    split_path = Path(config["paths"]["output"]) / "splits.csv"
    split_path.write_text(split_path.read_text().replace(",train", ",val"))  # edit the saved split
    prepare_dataset(config, log=lambda *_: None)
    assert set(load_split(split_path).values()) <= {"val", "test"}  # reused, not remade

    Image.new("RGB", (100, 100)).save(Path(config["paths"]["images"]) / "999_left.jpeg")
    write_csv(Path(config["paths"]["annotations"]) / "new.csv", [row("999_left.jpeg", CLASSES[0], 1, 1, 5, 5)])
    with pytest.raises(RuntimeError, match="resplit"):
        prepare_dataset(config, log=lambda *_: None)
    prepare_dataset(config, resplit=True, log=lambda *_: None)


def test_split_balances_classes():
    rng = np.random.default_rng(0)
    probs = np.array([10, 5, 4, 2] + [1] * (len(CLASSES) - 4), dtype=float)
    records = []
    for patient in range(600):
        for eye in ("left", "right"):
            boxes = [Box(CLASSES[rng.choice(len(CLASSES), p=probs / probs.sum())], 0, 0, 5, 5)
                     for _ in range(rng.integers(1, 12))]
            records.append(ImageRecord(Path(f"{patient}_{eye}.jpeg"), 100, 100, str(patient), None, boxes))
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    assignment = assign_splits(records, CLASSES, ratios)
    for label in CLASSES:
        counts = {s: sum(b.label == label for r in records if assignment[r.path.name] == s for b in r.boxes)
                  for s in SPLITS}
        for split in SPLITS:
            assert abs(counts[split] / sum(counts.values()) - ratios[split]) < 0.03, (label, counts)


def test_train_only_uses_allowed_gpus(monkeypatch):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from train import restrict_gpus

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    assert restrict_gpus(None, [2, 3]) == "0,1"  # default: both allowed GPUs
    assert restrict_gpus("3", [2, 3]) == "1"     # GPU 3 is torch device 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"  # GPUs 0 and 1 are hidden
    for bad in ("0", "1", "2,1", "2,2", "cpu"):
        with pytest.raises(SystemExit):
            restrict_gpus(bad, [2, 3])
