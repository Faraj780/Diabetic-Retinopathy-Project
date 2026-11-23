"""Write the dataset in the layout Ultralytics YOLO expects:

    yolo/images/train/10_left.jpeg   symlink to the original image (no copy, nothing written to it)
    yolo/labels/train/10_left.txt    one line per box: class cx cy w h, all 0-1 relative to the image size
    ...val/, test/
    yolo/dataset.yaml
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from .annotations import Box, ImageRecord


def to_yolo_line(box: Box, class_id: int, width: int, height: int) -> str:
    cx = (box.x1 + box.x2) / 2 / width
    cy = (box.y1 + box.y2) / 2 / height
    return f"{class_id} {cx:.6f} {cy:.6f} {(box.x2 - box.x1) / width:.6f} {(box.y2 - box.y1) / height:.6f}"


def export_yolo(records: list[ImageRecord], assignment: dict[str, str], class_names: list[str], out_dir: Path) -> Path:
    """Rebuild out_dir from scratch and return the path of dataset.yaml."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        # Removes the symlinks, never the images they point to. Also clears Ultralytics' *.cache files.
        shutil.rmtree(out_dir)
    class_id = {name: i for i, name in enumerate(class_names)}

    for record in records:
        split = assignment[record.path.name]
        image_dir, label_dir = out_dir / "images" / split, out_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / record.path.name).symlink_to(record.path.resolve())
        lines = [to_yolo_line(box, class_id[box.label], record.width, record.height) for box in record.boxes]
        (label_dir / f"{record.path.stem}.txt").write_text("\n".join(lines) + "\n")

    dataset = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(class_names)),
    }
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True))
    return yaml_path
