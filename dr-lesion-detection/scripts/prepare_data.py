#!/usr/bin/env python3
"""Turn the annotation CSVs into a train/val/test split and a YOLO dataset.

    python scripts/prepare_data.py
    python scripts/prepare_data.py --annotations /mnt/hdd2/faraj/BBoxAnnotations --images /mnt/hdd2/faraj/train

The split is saved to <output>/splits.csv and reused on later runs, so every model is
trained and tested on the same images. Pass --resplit only if you mean to change it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drdet.config import load_config  # noqa: E402
from drdet.pipeline import SplitMismatchError, prepare_dataset  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="config file (default: configs/default.yaml)")
    parser.add_argument("--annotations", type=Path, help="folder (or single file) of annotation CSVs")
    parser.add_argument("--images", type=Path, help="folder of fundus images")
    parser.add_argument("--grades", type=Path, help="Kaggle trainLabels.csv with image-level DR grades")
    parser.add_argument("--output", type=Path, help="where to write the split and the YOLO data")
    parser.add_argument("--resplit", action="store_true", help="make a new split even if splits.csv exists")
    parser.add_argument("--seed", type=int, help="split seed (only used with a new split)")
    parser.add_argument("--link-mode", choices=["symlink", "copy", "hardlink"], help="how images are placed in the YOLO folders")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    try:
        prepare_dataset(
            config,
            annotations=args.annotations,
            images=args.images,
            grades_csv=args.grades,
            output=args.output,
            resplit=args.resplit,
            seed=args.seed,
            link_mode=args.link_mode,
        )
    except (FileNotFoundError, FileExistsError, SplitMismatchError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Next: python scripts/check_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
