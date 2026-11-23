"""Split images into train / val / test.

Each patient's left and right eye go in the same split (the two eyes look alike, so splitting
them would leak training information into testing). Each split also gets about its share of
the images, of the boxes of every lesion class and of the images of every DR grade.

How: iterative stratification (Sechidis, Tsoumakas & Vlahavas, 2011) over patients. Take the
rarest feature (e.g. NVE boxes) and give each patient that has it to the split that still
needs the most of it; repeat until every patient is placed. This is tried RESTARTS times with
different random orders and the most balanced result is kept.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .annotations import ImageRecord

SPLITS = ("train", "val", "test")
RESTARTS = 20


def _stratify_once(counts: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One pass. counts: patients x features (column 0 = images). Returns a split index per patient."""
    need = target[:, None] * counts.sum(axis=0)[None, :]  # how much of each feature each split still wants
    unassigned = np.ones(len(counts), dtype=bool)
    choice = np.full(len(counts), -1)
    while unassigned.any():
        remaining = counts[unassigned].sum(axis=0)
        features = np.flatnonzero(remaining > 0)
        feature = features[np.lexsort((rng.random(len(features)), remaining[features]))[0]]  # rarest
        patients = np.flatnonzero(unassigned & (counts[:, feature] > 0))
        rng.shuffle(patients)
        for patient in patients:
            # The split that needs this feature most (ties: needs most images, then random).
            split = np.lexsort((rng.random(len(target)), -need[:, 0], -need[:, feature]))[0]
            choice[patient] = split
            need[split] -= counts[patient]
            unassigned[patient] = False
    return choice


def assign_splits(records: list[ImageRecord], class_names: list[str], ratios: dict, seed: int = 0) -> dict[str, str]:
    """Return {image file name: split}."""
    target = np.array([ratios[split] for split in SPLITS])
    patients = sorted({r.patient for r in records})
    row_of = {patient: i for i, patient in enumerate(patients)}
    grades = sorted({r.grade for r in records if r.grade is not None})
    column_of = {name: i for i, name in enumerate(["images", *class_names, *(f"grade {g}" for g in grades)])}

    counts = np.zeros((len(patients), len(column_of)))
    for record in records:
        row = row_of[record.patient]
        counts[row, 0] += 1
        for box in record.boxes:
            counts[row, column_of[box.label]] += 1
        if record.grade is not None:
            counts[row, column_of[f"grade {record.grade}"]] += 1
    counts = counts[:, counts.sum(axis=0) > 0]  # ignore classes that never occur
    weights = np.ones(counts.shape[1])
    weights[0] = 4.0  # getting the split sizes right matters most

    best_cost, best_choice = None, None
    for attempt in range(RESTARTS):
        choice = _stratify_once(counts, target, np.random.default_rng([seed, attempt]))
        shares = np.stack([counts[choice == s].sum(axis=0) for s in range(len(SPLITS))]) / counts.sum(axis=0)
        cost = float((((shares - target[:, None]) ** 2) @ weights).sum())
        if best_cost is None or cost < best_cost:
            best_cost, best_choice = cost, choice
    return {r.path.name: SPLITS[best_choice[row_of[r.patient]]] for r in records}


def save_split(path: Path, records: list[ImageRecord], assignment: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_file", "patient_id", "split"])
        for record in records:
            writer.writerow([record.path.name, record.patient, assignment[record.path.name]])


def write_split_summary(path: Path, records: list[ImageRecord], assignment: dict, class_names: list[str]) -> None:
    """CSV showing how balanced the split is: one row per thing counted (images, patients, boxes
    of each class, images of each DR grade) with its number and percentage in each split."""
    counts = defaultdict(Counter)  # row name -> {split: count}
    patients = defaultdict(set)
    for record in records:
        split = assignment[record.path.name]
        counts["images"][split] += 1
        patients[split].add(record.patient)
        counts["boxes (all classes)"][split] += len(record.boxes)
        for box in record.boxes:
            counts[f"boxes: {box.label}"][split] += 1
        grade = "unknown" if record.grade is None else record.grade
        counts[f"images with DR grade {grade}"][split] += 1
    counts["patients"] = Counter({split: len(ids) for split, ids in patients.items()})

    order = ["images", "patients", "boxes (all classes)", *(f"boxes: {name}" for name in class_names),
             *sorted(name for name in counts if name.startswith("images with DR grade"))]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["count", "total", *SPLITS, *(f"{split} %" for split in SPLITS)])
        for name in order:
            per_split = counts[name]
            total = sum(per_split.values())
            percents = [round(100 * per_split[split] / total, 1) if total else 0 for split in SPLITS]
            writer.writerow([name, total, *(per_split[split] for split in SPLITS), *percents])


def load_split(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["image_file"]: row["split"] for row in csv.DictReader(handle)}
