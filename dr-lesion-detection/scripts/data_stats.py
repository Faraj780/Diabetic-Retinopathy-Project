#!/usr/bin/env python3
"""Charts and a table of the prepared dataset: class counts, split balance, box sizes.

    python scripts/data_stats.py      # writes to outputs/stats/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drdet.config import load_config, resolve_path  # noqa: E402
from drdet.stats import compute_stats, write_stats  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, help="the folder prepare_data.py wrote to")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/stats"))
    args = parser.parse_args(argv)

    config = load_config(args.config)
    output = resolve_path(args.output or config["paths"].get("output", "data/processed"))
    yolo_dir = output / "yolo"
    if not (yolo_dir / "dataset.yaml").exists():
        print(f"ERROR: {yolo_dir}/dataset.yaml not found. Run prepare_data.py first.", file=sys.stderr)
        return 1
    class_names = list(config["classes"])
    ratios = config.get("split", {}).get("ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    stats = compute_stats(yolo_dir, class_names)
    for path in write_stats(stats, class_names, ratios, resolve_path(args.out_dir)):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
