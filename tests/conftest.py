"""A small fake dataset in the same CSV layout as the real annotations."""

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
WIDTH, HEIGHT = 400, 300


def row(image, label, left=0.0, top=0.0, width=0.0, height=0.0, scale=1.0, kind="rectangle"):
    """One CSV row. Coordinates are given in original-image pixels and stored scaled, as the tool does."""
    return {"image_name": image, "annotation_type": kind, "label": label,
            "left_coordinate": left * scale, "top_coordinate": top * scale, "width": width * scale,
            "height": height * scale, "image_scale_x": scale, "image_scale_y": scale}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def config(tmp_path):
    """The real config, pointed at a fake dataset of 30 patients (some with one eye only)."""
    rng = random.Random(0)
    images, annotations = tmp_path / "train", tmp_path / "annotations"
    images.mkdir()
    rows = []
    for patient in range(1, 31):
        for eye in (["left", "right"] if patient % 5 else ["left"]):
            name = f"{patient}_{eye}.jpeg"
            Image.new("RGB", (WIDTH, HEIGHT)).save(images / name)
            rows.append(row(name, "Moderate Non-Proliferative Diabetic Retinopathy", kind="whole_image"))
            for _ in range(rng.randint(1, 5)):
                w, h = rng.uniform(4, 40), rng.uniform(4, 40)
                rows.append(row(name, rng.choice(CLASSES), rng.uniform(0, WIDTH - w), rng.uniform(0, HEIGHT - h),
                                w, h, scale=rng.choice([1.0, 0.25])))
    write_csv(annotations / "batch.csv", rows)

    config = load_config()
    config["paths"] = {"annotations": str(annotations), "images": str(images), "output": str(tmp_path / "processed"),
                       "results": str(tmp_path / "outputs")}
    return config
