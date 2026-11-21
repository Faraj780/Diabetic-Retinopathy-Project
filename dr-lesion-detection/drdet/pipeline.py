"""The full preparation run: CSVs -> clean records -> split -> YOLO dataset."""

from __future__ import annotations

import json

from .annotations import build_records, load_grades
from .config import resolve_path
from .export_yolo import export_yolo
from .split import assign_splits, format_split_table, load_split_file, save_split_file, split_table


class SplitMismatchError(RuntimeError):
    pass


def prepare_dataset(
    config: dict,
    *,
    annotations=None,
    images=None,
    grades_csv=None,
    output=None,
    resplit: bool = False,
    seed: int | None = None,
    link_mode: str | None = None,
    log=print,
) -> dict:
    paths = config.get("paths", {})
    annotations = resolve_path(annotations or paths.get("annotations"))
    images = resolve_path(images or paths.get("images"))
    grades_csv = resolve_path(grades_csv or paths.get("grades_csv"))
    output = resolve_path(output or paths.get("output") or "data/processed")
    class_names = list(config["classes"])
    split_cfg = config.get("split", {})
    cleaning = config.get("cleaning", {})
    csv_cfg = config.get("csv", {})
    link_mode = link_mode or config.get("export", {}).get("link_mode", "symlink")
    seed = split_cfg.get("seed", 0) if seed is None else seed

    for label, path in (("annotations", annotations), ("images", images)):
        if path is None or not path.exists():
            raise FileNotFoundError(f"{label} path not found: {path}. Set it in the config or pass --{label}.")

    grades = None
    if grades_csv is not None:
        if grades_csv.exists():
            grades = load_grades(grades_csv)
            log(f"Read {len(grades)} DR grades from {grades_csv}")
        else:
            log(f"WARNING: grades file {grades_csv} not found; splitting without DR grades.")

    log(f"Reading annotations from {annotations}")
    records, report = build_records(
        annotations,
        images,
        class_names,
        columns=csv_cfg.get("columns"),
        rectangle_types=csv_cfg.get("rectangle_types", ["rectangle"]),
        patient_regex=split_cfg.get("patient_regex"),
        grades=grades,
        min_box_size=float(cleaning.get("min_box_size_px", 1.0)),
        keep_images_without_boxes=bool(cleaning.get("keep_images_without_boxes", False)),
    )
    log(summarise_report(report))
    if not records:
        raise RuntimeError("No usable images were found. Check the paths and the CSV column names in the config.")

    output.mkdir(parents=True, exist_ok=True)
    split_path = output / "splits.csv"
    if split_path.exists() and not resplit:
        saved = load_split_file(split_path)
        missing = [r.path.name for r in records if r.path.name not in saved]
        if missing:
            raise SplitMismatchError(
                f"{len(missing)} images (e.g. {missing[:3]}) aren't in the saved split {split_path}. "
                "Rerun with --resplit to make a new split. Results from older runs then aren't comparable."
            )
        unused = len(set(saved) - {r.path.name for r in records})
        if unused:
            log(f"Note: {unused} images in the saved split are no longer kept by the cleaning step.")
        assignment = {r.path.name: saved[r.path.name] for r in records}
        split_info = {"method": f"reused {split_path.name}"}
        log(f"Reusing the saved split in {split_path} (pass --resplit to make a new one).")
    else:
        assignment, split_info = assign_splits(
            records,
            class_names,
            split_cfg.get("ratios", {"train": 0.7, "val": 0.15, "test": 0.15}),
            seed=seed,
            restarts=int(split_cfg.get("restarts", 20)),
            group_by_patient=bool(split_cfg.get("group_by_patient", True)),
        )
        save_split_file(split_path, records, assignment)
        log(f"Made a new split ({split_info['method']}, seed {seed}) and saved it to {split_path}")

    ratios = split_cfg.get("ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    table = split_table(records, assignment, class_names)
    table_text = format_split_table(table, ratios)
    log("\n" + table_text + "\n")

    log("Writing YOLO dataset...")
    yolo = export_yolo(records, assignment, class_names, output / "yolo", link_mode)

    report["class_order"] = class_names
    report["split"] = {**split_info, "ratios": ratios, "table": table}
    report["outputs"] = {"splits_csv": str(split_path), "yolo": yolo}
    (output / "prepare_report.json").write_text(json.dumps(report, indent=2))
    (output / "split_report.md").write_text(f"# Split balance\n\n{split_info.get('method')}\n\n{table_text}\n")
    log(f"Done. Dataset YAML for YOLO: {yolo['dataset_yaml']}")
    return report


def summarise_report(report: dict) -> str:
    dropped = report["boxes_dropped"]
    lines = [
        f"CSV files: {report['csv_files']}, rows: {report['csv_rows']} "
        f"(exact duplicates removed: {report['duplicate_csv_rows']})",
        f"Annotation types: {report['annotation_types']}",
        f"Images named in CSVs: {report['images_in_csv']}, kept: {report['images_kept']}, "
        f"missing file: {report['images_missing_file']}, unreadable: {report['images_unreadable']}, "
        f"no usable boxes: {report['images_without_boxes']}",
        f"Patients: {report['patients']} ({report['patients_with_both_eyes']} with both eyes); "
        f"file names not matching the patient pattern: {report['images_unmatched_patient_pattern']}",
        f"Images with a DR grade: {report['images_with_grade']}",
        f"Rectangle rows: {report['rectangle_rows']} -> kept {report['boxes_kept']}, dropped {sum(dropped.values())} {dropped}",
        f"Kept but adjusted: {report['boxes_adjusted']}",
        "Boxes per class: " + ", ".join(f"{k}: {v}" for k, v in report["boxes_kept_per_class"].items()),
    ]
    if report["missing_file_examples"]:
        lines.append(f"Examples of missing image files: {report['missing_file_examples']}")
    if report["unknown_labels"]:
        lines.append(f"WARNING: labels not in the config's class list were dropped: {report['unknown_labels']}")
    return "\n".join(lines)
