#!/usr/bin/env python3
"""Turn the annotation CSVs into a train/val/test split and a YOLO dataset (paths in the config).

    python scripts/prepare_data.py
    python scripts/prepare_data.py --resplit   # new split: only on purpose, old runs stop being comparable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drdet.config import load_config  # noqa: E402
from drdet.pipeline import prepare_dataset  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="config file (default: configs/default.yaml)")
    parser.add_argument("--resplit", action="store_true", help="make a new split even if splits.csv exists")
    args = parser.parse_args(argv)
    try:
        yaml_path = prepare_dataset(load_config(args.config), resplit=args.resplit)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Done. Dataset for YOLO: {yaml_path}\nNext: python scripts/train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
