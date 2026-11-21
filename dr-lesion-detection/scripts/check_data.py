#!/usr/bin/env python3
"""Check the prepared YOLO data before training. Exits with code 1 if any check fails.

    python scripts/check_data.py
    python scripts/check_data.py --open-images   # also open every image (slower)

By default the CSVs are parsed again and every label file is compared with them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drdet.annotations import build_records  # noqa: E402
from drdet.checks import run_checks  # noqa: E402
from drdet.config import load_config, resolve_path  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--annotations", type=Path, help="annotation CSVs (default: from the config)")
    parser.add_argument("--images", type=Path, help="original images (default: from the config)")
    parser.add_argument("--output", type=Path, help="the folder prepare_data.py wrote to")
    parser.add_argument("--tolerance", type=float, default=1.0, help="allowed label/CSV difference in pixels")
    parser.add_argument("--open-images", action="store_true", help="open every image to confirm it isn't corrupt")
    parser.add_argument("--skip-csv-compare", action="store_true", help="don't re-read the CSVs")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    class_names = list(config["classes"])
    output = resolve_path(args.output or config["paths"].get("output", "data/processed"))

    records = None
    if not args.skip_csv_compare:
        csv_cfg, cleaning = config.get("csv", {}), config.get("cleaning", {})
        records, _ = build_records(
            resolve_path(args.annotations or config["paths"]["annotations"]),
            resolve_path(args.images or config["paths"]["images"]),
            class_names,
            columns=csv_cfg.get("columns"),
            rectangle_types=csv_cfg.get("rectangle_types", ["rectangle"]),
            patient_regex=config.get("split", {}).get("patient_regex"),
            min_box_size=float(cleaning.get("min_box_size_px", 1.0)),
            keep_images_without_boxes=bool(cleaning.get("keep_images_without_boxes", False)),
        )

    failed = 0
    for result in run_checks(output, class_names, records, args.tolerance, args.open_images):
        if result.passed:
            tag = "PASS"
        elif result.level == "warning":
            tag = "WARN"
        else:
            tag = "FAIL"
            failed += 1
        print(f"[{tag}] {result.name}: {result.detail}")
    print(f"\n{failed} check(s) failed." if failed else "\nAll checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
