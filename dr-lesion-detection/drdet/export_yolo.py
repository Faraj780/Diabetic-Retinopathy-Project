"""Write the dataset in the folder layout Ultralytics YOLO expects.

    yolo/
      images/train/10_left.jpeg   (symlink or copy of the original)
      labels/train/10_left.txt    one line per box: class cx cy w h  (all 0-1, relative to image size)
      ...val/, test/
      dataset.yaml

YOLO finds a label file by swapping "images" for "labels" in the image path.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import yaml

from .annotations import Box, ImageRecord
from .split import SPLITS

MARKER_NAME = ".drdet_export"


def reset_output_dir(path: Path) -> None:
    """Empty a folder this project created; refuse to touch anything else."""
    path = Path(path)
    if path.exists():
        if (path / MARKER_NAME).exists():
            shutil.rmtree(path)  # removes symlinks, never the images they point to
        elif any(path.iterdir()):
            raise FileExistsError(
                f"{path} already has files that weren't created by this project. "
                "Move them or choose another output folder."
            )
    path.mkdir(parents=True, exist_ok=True)
    (path / MARKER_NAME).write_text("Created by drdet. Safe to delete: prepare_data.py rebuilds it.\n")


def to_yolo_line(box: Box, class_id: int, image_width: int, image_height: int) -> str:
    cx = (box.x1 + box.x2) / 2 / image_width
    cy = (box.y1 + box.y2) / 2 / image_height
    w = box.width / image_width
    h = box.height / image_height
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def parse_yolo_line(line: str, image_width: int, image_height: int):
    """Inverse of to_yolo_line. Also accepts a 6th value (confidence) from prediction files.

    Returns (class_id, (x1, y1, x2, y2) in pixels, confidence or None).
    """
    parts = line.split()
    if len(parts) not in (5, 6):
        raise ValueError(f"expected 5 or 6 values, got {len(parts)}: {line!r}")
    class_id = int(parts[0])
    cx, cy, w, h = (float(v) for v in parts[1:5])
    confidence = float(parts[5]) if len(parts) == 6 else None
    x1, x2 = (cx - w / 2) * image_width, (cx + w / 2) * image_width
    y1, y2 = (cy - h / 2) * image_height, (cy + h / 2) * image_height
    return class_id, (x1, y1, x2, y2), confidence


def read_label_file(path: Path, image_width: int, image_height: int, class_names: list[str]):
    """Read a YOLO label (or prediction) file into [(Box, confidence)]."""
    boxes = []
    if not Path(path).exists():
        return boxes
    for line in Path(path).read_text().splitlines():
        if line.strip():
            class_id, (x1, y1, x2, y2), confidence = parse_yolo_line(line, image_width, image_height)
            label = class_names[class_id] if 0 <= class_id < len(class_names) else f"class {class_id}"
            boxes.append((Box(x1, y1, x2, y2, label), confidence))
    return boxes


def place_image(source: Path, destination: Path, mode: str) -> None:
    if mode == "symlink":
        destination.symlink_to(Path(source).resolve())
    elif mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        raise ValueError(f"link_mode must be symlink, copy or hardlink, not {mode!r}")


def write_dataset_yaml(out_dir: Path, class_names: list[str], train="images/train", val="images/val", test="images/test") -> Path:
    data = {
        "path": str(Path(out_dir).resolve()),
        "train": train,
        "val": val,
        "test": test,
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }
    yaml_path = Path(out_dir) / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    return yaml_path


def export_yolo(records: list[ImageRecord], assignment: dict, class_names: list[str], out_dir: Path, link_mode: str = "symlink") -> dict:
    stems = [r.stem for r in records]
    duplicates = sorted({s for s in stems if stems.count(s) > 1}) if len(set(stems)) != len(stems) else []
    if duplicates:
        raise ValueError(f"Two images share a name (label files would clash): {duplicates[:5]}")

    out_dir = Path(out_dir)
    reset_output_dir(out_dir)  # also deletes stale Ultralytics *.cache files
    for split in SPLITS:
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)

    class_ids = {name: i for i, name in enumerate(class_names)}
    counts = {split: {"images": 0, "boxes": 0} for split in SPLITS}
    for record in records:
        split = assignment[record.path.name]
        place_image(record.path, out_dir / "images" / split / record.path.name, link_mode)
        lines = [to_yolo_line(b, class_ids[b.label], record.width, record.height) for b in record.boxes]
        (out_dir / "labels" / split / f"{record.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        counts[split]["images"] += 1
        counts[split]["boxes"] += len(lines)

    yaml_path = write_dataset_yaml(out_dir, class_names)
    return {"dataset_yaml": str(yaml_path), "counts": counts, "link_mode": link_mode}


def make_subset_dataset(yolo_dir: Path, out_dir: Path, n_images: int, class_names: list[str], seed: int = 0, split: str = "train") -> Path:
    """Tiny dataset for the overfit sanity check: n images, used for both training and validation.

    A working pipeline should nearly memorise them. If it can't, suspect the data or labels.
    """
    yolo_dir, out_dir = Path(yolo_dir), Path(out_dir)
    labelled = []
    for image in sorted((yolo_dir / "images" / split).iterdir()):
        label = yolo_dir / "labels" / split / f"{image.stem}.txt"
        if not image.name.startswith(".") and label.exists() and label.read_text().strip():
            labelled.append((image, label))
    if not labelled:
        raise FileNotFoundError(f"No labelled images in {yolo_dir / 'images' / split}. Run prepare_data.py first.")
    chosen = random.Random(seed).sample(labelled, min(n_images, len(labelled)))

    reset_output_dir(out_dir)
    (out_dir / "images" / "train").mkdir(parents=True)
    (out_dir / "labels" / "train").mkdir(parents=True)
    for image, label in chosen:
        (out_dir / "images" / "train" / image.name).symlink_to(image.resolve())
        shutil.copy2(label, out_dir / "labels" / "train" / label.name)
    return write_dataset_yaml(out_dir, class_names, train="images/train", val="images/train", test="images/train")
