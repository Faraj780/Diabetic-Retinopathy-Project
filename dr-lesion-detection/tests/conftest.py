"""A small fake dataset with the same CSV layout as the real annotations, plus edge cases."""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from drdet.config import load_config  # noqa: E402

CLASSES = load_config()["classes"]
COLUMNS = ["image_name", "annotation_type", "label", "left_coordinate", "top_coordinate",
           "width", "height", "image_scale_x", "image_scale_y", "annotator"]
WIDTH, HEIGHT = 400, 300


def row(image, label, left, top, w, h, scale=1.0, kind="rectangle", annotator="a"):
    return {"image_name": image, "annotation_type": kind, "label": label,
            "left_coordinate": left * scale, "top_coordinate": top * scale, "width": w * scale,
            "height": h * scale, "image_scale_x": scale, "image_scale_y": scale, "annotator": annotator}


def make_fake_dataset(root: Path, patients: int = 24, seed: int = 0) -> dict:
    """Write images, two annotation CSVs and a grades file. Returns paths and expected counts."""
    rng = random.Random(seed)
    images_dir, ann_dir = root / "train", root / "BBoxAnnotations"
    images_dir.mkdir(parents=True)
    (ann_dir / "batch2").mkdir(parents=True)
    rows_a, rows_b, grades = [], [], []
    good_boxes = 0

    for patient in range(1, patients + 1):
        eyes = ["left", "right"] if patient % 5 else ["left"]  # every 5th patient has one eye only
        for eye in eyes:
            name = f"{patient}_{eye}.jpeg"
            Image.new("RGB", (WIDTH, HEIGHT), (120, 40, 20)).save(images_dir / name)
            grades.append({"image": f"{patient}_{eye}", "level": rng.randint(1, 4)})
            for _ in range(rng.randint(1, 6)):
                label = CLASSES[0] if rng.random() < 0.5 else rng.choice(CLASSES)  # microaneurysms are common
                w, h = rng.uniform(4, 40), rng.uniform(4, 40)
                x, y = rng.uniform(0, WIDTH - w), rng.uniform(0, HEIGHT - h)
                scale = 2.0 if patient % 3 == 0 else 1.0  # some images annotated on a resized view
                rows_a.append(row(name, label, x, y, w, h, scale))
                good_boxes += 1

    edge = "1_left.jpeg"
    rows_a.append(dict(rows_a[0]))                                         # exact duplicate row
    dup = dict(rows_a[0])
    dup["annotator"] = "b"
    rows_b.append(dup)                                                     # same box, different annotator
    rows_b.append(row(edge, "Not A Real Lesion", 10, 10, 20, 20))          # unknown label
    rows_b.append(row(edge, CLASSES[1], 10, 10, 0, 20))                    # zero width
    rows_b.append(row(edge, CLASSES[1], 500, 500, 10, 10))                 # outside image
    rows_b.append(row(edge, CLASSES[2], 390, 290, 30, 30))                 # hangs off the corner: clipped, kept
    rows_b.append(row(edge, CLASSES[3], 60, 60, -20, 15))                  # negative width: flipped, kept
    rows_b.append(row(edge, CLASSES[1], 10, 10, 5, 5, kind="polygon"))     # not a rectangle
    rows_b.append(row("999_left.jpeg", CLASSES[0], 10, 10, 5, 5))          # image file missing
    good_boxes += 2

    for path, rows in ((ann_dir / "batch1.csv", rows_a), (ann_dir / "batch2" / "batch2.csv", rows_b)):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    with open(root / "trainLabels.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "level"])
        writer.writeheader()
        writer.writerows(grades)

    return {"images": images_dir, "annotations": ann_dir, "grades": root / "trainLabels.csv",
            "good_boxes": good_boxes, "rectangle_rows": len(rows_a) - 1 + len(rows_b) - 1}


@pytest.fixture
def fake(tmp_path):
    data = make_fake_dataset(tmp_path / "raw")
    data["output"] = tmp_path / "processed"
    return data


@pytest.fixture
def config():
    return load_config()
