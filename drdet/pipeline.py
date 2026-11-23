"""Data preparation: CSVs -> boxes per image -> train/val/test split -> YOLO dataset."""

from __future__ import annotations

from pathlib import Path

from .annotations import build_records
from .config import resolve_path
from .export_yolo import export_yolo
from .split import SPLITS, assign_splits, load_split, save_split, write_split_summary


def prepare_dataset(config: dict, resplit: bool = False, log=print) -> Path:
    """Build the YOLO dataset and return the path of its dataset.yaml.

    The split is saved to <output>/splits.csv and reused on later runs, so every model is
    trained and tested on the same images. resplit=True makes a new one.
    """
    paths = config["paths"]
    output = resolve_path(paths["output"])
    class_names = list(config["classes"])

    records, dropped = build_records(resolve_path(paths["annotations"]), resolve_path(paths["images"]),
                                     class_names, config.get("label_map", {}), config.get("grade_labels", {}))
    if not records:
        raise RuntimeError("No images with usable boxes were found. Check the paths in the config.")
    log(f"Images: {len(records)}, boxes: {sum(len(r.boxes) for r in records)}, boxes dropped: {dict(dropped)}")

    output.mkdir(parents=True, exist_ok=True)
    split_path = output / "splits.csv"
    if split_path.exists() and not resplit:
        assignment = load_split(split_path)
        missing = [r.path.name for r in records if r.path.name not in assignment]
        if missing:
            raise RuntimeError(f"{len(missing)} images (e.g. {missing[:3]}) aren't in {split_path}. Rerun with "
                               "--resplit to make a new split (results of earlier runs then aren't comparable).")
        log(f"Using the saved split {split_path} (--resplit makes a new one)")
    else:
        split_cfg = config["split"]
        assignment = assign_splits(records, class_names, split_cfg["ratios"], split_cfg["seed"])
        save_split(split_path, records, assignment)
        log(f"Made a new split and saved it to {split_path}")

    for split in SPLITS:
        in_split = [r for r in records if assignment[r.path.name] == split]
        log(f"  {split}: {len(in_split)} images, {sum(len(r.boxes) for r in in_split)} boxes")
    results = resolve_path(paths["results"])
    results.mkdir(parents=True, exist_ok=True)
    write_split_summary(results / "split_summary.csv", records, assignment, class_names)
    log(f"Split balance per class and grade: {results / 'split_summary.csv'}")
    return export_yolo(records, assignment, class_names, output / "yolo")
