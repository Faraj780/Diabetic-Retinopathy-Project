# Project notes for Claude Code

Final-year project: benchmark object detectors for diabetic retinopathy lesion detection on
Kaggle fundus images with bounding-box annotations. Supervisor: Chaitanya Kaul.
See README.md for the workflow and roadmap.

## Current stage: YOLO format and YOLOv8 only

- The data pipeline exports YOLO format only, and `scripts/train.py` supports YOLOv8 only.
- Don't add COCO export, other data formats or other models (YOLOv5, YOLO26, Faster R-CNN,
  DETR) until the student says YOLOv8 testing is complete. The README roadmap lists the order.
- Focus help on getting YOLOv8 working on this data: data checks, settings, reading results.

## Rules from the supervisor

- The student must understand every change. Explain what you changed and why, in plain
  terms, and prefer small, readable code over clever code. Design decisions (architectures,
  how data is fed) are the student's to make: suggest options, don't decide silently.
- Don't modify model architectures or loss functions. Models are compared as published;
  only the data loading and training wrappers are ours.
- The goal is a fair benchmark, not a high score. Later models must use the same split,
  image size and evaluation as YOLOv8.

## Data rules

- `data/processed/splits.csv` is the fixed split. Never regenerate it (`--resplit`) unless
  the student explicitly asks; it invalidates earlier runs.
- The test split is used once per final model, at the end. Never tune on it or use it
  during training.
- All CSV parsing goes through `drdet/annotations.py`. Fix parsing there, not in a script.
- After changing anything in the data pipeline, run `python -m pytest -q`, then
  `python scripts/prepare_data.py` and `python scripts/check_data.py`.

## Server

- Shared GPU server; data lives under `/mnt/hdd2/faraj`. Check `nvidia-smi` and use a free GPU.
- The student has sudo but was asked not to use it beyond creating his directory. Don't run
  sudo, install system packages or change anything outside the project and its venv.
- Run long training inside tmux.

## Commands

```bash
python -m pytest -q                                  # tests (fake data, no GPU needed)
python scripts/prepare_data.py                       # split + YOLO export
python scripts/check_data.py                         # data checks (re-reads the CSVs)
python scripts/visualize.py --source yolo --split train --zoom
python scripts/data_stats.py
python scripts/train.py --dry-run                    # show settings
python scripts/train.py --overfit 10 --epochs 100    # sanity check
python scripts/train.py --size n --imgsz 1024 --epochs 50
```

## Code layout

- `drdet/` library code; `scripts/` thin command-line wrappers; `tests/` pytest.
- Class order in `configs/default.yaml` defines YOLO class ids; don't reorder it.
