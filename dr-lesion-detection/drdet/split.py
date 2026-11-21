"""Split images into train / val / test.

Two rules, from the supervisor meetings:
  * Keep each patient's left and right eye in the same split (the two eyes look alike,
    so splitting them would leak information from training into testing).
  * Make each split hold roughly its share (e.g. 70/15/15) of everything that matters:
    images, left eyes, right eyes, boxes of each lesion type and images of each DR grade.

How it works: iterative stratification (Sechidis, Tsoumakas & Vlahavas, 2011), applied to
patients instead of single images.
  1. Describe each patient by counts: images, left eyes, right eyes, boxes of each lesion
     type, images of each DR grade. Each split "needs" its ratio times every total.
  2. Take the rarest feature that still has unassigned patients (e.g. NVE boxes). Give each
     patient that has it to the split that still needs the most of that feature (ties: the
     split that needs the most images). Subtract the patient's counts from that split's needs.
  3. Repeat until every patient is assigned. Rare lesions are placed first, while every
     split can still take its share; common ones even out at the end.
  4. Do this several times with shuffled orders and keep the most balanced result.
Perfect balance is usually impossible (the data is imbalanced), so check the table it prints.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .annotations import ImageRecord

SPLITS = ("train", "val", "test")

# How much each kind of feature matters in the balance score. Images weigh more so
# the split sizes come out close to the requested ratios.
DEFAULT_WEIGHTS = {"images": 4.0, "eye": 1.0, "class": 1.0, "grade": 1.0}


def check_ratios(ratios: dict) -> np.ndarray:
    missing = [s for s in SPLITS if s not in ratios]
    if missing:
        raise ValueError(f"split ratios are missing {missing}")
    target = np.array([float(ratios[s]) for s in SPLITS])
    if (target <= 0).any() or abs(target.sum() - 1.0) > 1e-6:
        raise ValueError(f"split ratios must be positive and add up to 1, got {ratios}")
    return target


def group_features(records: list[ImageRecord], class_names: list[str], group_by_patient: bool = True):
    """Build one row of counts per group (patient, or single image if not grouping)."""
    members: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        members[record.patient_id if group_by_patient else record.path.name].append(index)

    grades = sorted({r.grade for r in records if r.grade is not None})
    names = ["images", "eye:left", "eye:right"]
    names += [f"class:{c}" for c in class_names] + [f"grade:{g}" for g in grades]
    column = {name: i for i, name in enumerate(names)}

    matrix = np.zeros((len(members), len(names)))
    for row, indices in enumerate(members.values()):
        for index in indices:
            record = records[index]
            matrix[row, column["images"]] += 1
            if record.eye in ("left", "right"):
                matrix[row, column[f"eye:{record.eye}"]] += 1
            for box in record.boxes:
                matrix[row, column[f"class:{box.label}"]] += 1
            if record.grade is not None:
                matrix[row, column[f"grade:{record.grade}"]] += 1
    return list(members.keys()), list(members.values()), matrix, names


def _stratify_once(counts: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One pass of iterative stratification. counts: groups x features, column 0 = images."""
    n_groups = counts.shape[0]
    need = target[:, None] * counts.sum(axis=0)[None, :]  # splits x features still wanted
    unassigned = np.ones(n_groups, dtype=bool)
    choice = np.full(n_groups, -1)

    while unassigned.any():
        remaining = counts[unassigned].sum(axis=0)
        features = np.flatnonzero(remaining > 0)
        # Rarest remaining feature first (random among ties).
        feature = features[np.lexsort((rng.random(len(features)), remaining[features]))[0]]
        groups = np.flatnonzero(unassigned & (counts[:, feature] > 0))
        rng.shuffle(groups)
        for group in groups:
            # Most need for this feature, then most need for images, then random.
            split = np.lexsort((rng.random(len(target)), -need[:, 0], -need[:, feature]))[0]
            choice[group] = split
            need[split] -= counts[group]
            unassigned[group] = False
    return choice


def assign_splits(
    records: list[ImageRecord],
    class_names: list[str],
    ratios: dict,
    *,
    seed: int = 0,
    restarts: int = 20,
    group_by_patient: bool = True,
    weights: dict | None = None,
):
    """Return ({image file name: split}, info about the chosen split)."""
    target = check_ratios(ratios)
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    keys, members, counts, names = group_features(records, class_names, group_by_patient)

    totals = counts.sum(axis=0)
    used = totals > 0  # ignore classes or grades that never occur
    counts, totals = counts[:, used], totals[used]
    used_names = [n for n, u in zip(names, used) if u]
    w = np.array([weights[n.split(":")[0]] for n in used_names])

    best = None
    for attempt in range(max(1, restarts)):
        rng = np.random.default_rng([seed, attempt])
        choice = _stratify_once(counts, target, rng)
        # Score the result: how far each split's share of each feature is from its target.
        shares = np.stack([counts[choice == s].sum(axis=0) / totals for s in range(len(SPLITS))])
        cost = float((((shares - target[:, None]) ** 2) @ w).sum())
        if best is None or cost < best[0]:
            best = (cost, choice, attempt)

    cost, choice, attempt = best
    assignment = {}
    for group, indices in enumerate(members):
        for index in indices:
            assignment[records[index].path.name] = SPLITS[choice[group]]

    info = {
        "method": "iterative stratification by patient" if group_by_patient else "iterative stratification by image",
        "seed": seed,
        "restarts": restarts,
        "best_attempt": attempt,
        "imbalance_score": round(cost, 6),
        "groups": len(keys),
        "features_balanced": used_names,
    }
    return assignment, info


def split_table(records: list[ImageRecord], assignment: dict, class_names: list[str]) -> dict:
    """Counts per split for everything the split tries to balance."""
    table: dict[str, dict[str, int]] = {}

    def add(row: str, split: str, amount: int = 1):
        table.setdefault(row, {s: 0 for s in SPLITS})[split] += amount

    patients = {s: set() for s in SPLITS}
    for record in records:
        split = assignment[record.path.name]
        add("images", split)
        patients[split].add(record.patient_id)
        add(f"{record.eye} eyes", split)
        add("boxes (all)", split, len(record.boxes))
        if record.grade is not None:
            add(f"images with grade {record.grade}", split)
    for split in SPLITS:
        table.setdefault("patients", {s: 0 for s in SPLITS})[split] = len(patients[split])
    for name in class_names:
        table[f"boxes: {name}"] = {s: 0 for s in SPLITS}
    for record in records:
        split = assignment[record.path.name]
        for box in record.boxes:
            table[f"boxes: {box.label}"][split] += 1

    order = ["images", "patients", "left eyes", "right eyes", "unknown eyes", "boxes (all)"]
    order += [f"boxes: {n}" for n in class_names] + sorted(k for k in table if k.startswith("images with grade"))
    return {key: table[key] for key in order if key in table}


def format_split_table(table: dict, ratios: dict) -> str:
    """Markdown table: count and percentage of each row's total per split."""
    header = "| | total | " + " | ".join(f"{s} (target {ratios[s]:.0%})" for s in SPLITS) + " |"
    lines = [header, "|---|---:|" + "---:|" * len(SPLITS)]
    for row, counts in table.items():
        total = sum(counts.values())
        cells = [f"{counts[s]} ({counts[s] / total:.0%})" if total else "0" for s in SPLITS]
        lines.append(f"| {row} | {total} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_split_file(path: Path, records: list[ImageRecord], assignment: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_file", "image_name", "patient_id", "eye", "grade", "n_boxes", "split"])
        for r in records:
            grade = "" if r.grade is None else r.grade
            writer.writerow([r.path.name, r.image_name, r.patient_id, r.eye, grade, len(r.boxes), assignment[r.path.name]])


def load_split_file(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["image_file"]: row["split"] for row in csv.DictReader(handle)}
