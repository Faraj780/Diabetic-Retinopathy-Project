# Diabetic retinopathy lesion detection

Benchmark of object detectors on Kaggle DR fundus images with bounding-box lesion
annotations (7 lesion types). The aim, as agreed with the supervisor, is a working
benchmark that compares detector families on the same data, not a target accuracy.

**Current stage: YOLOv8 only.** The project prepares the data in YOLO format and trains
YOLOv8, to confirm the data pipeline works end to end. Other formats and models come
later (see [Roadmap](#roadmap)).

## Layout

```
configs/default.yaml     paths, class order, split settings
drdet/                   the code (imported by the scripts)
  annotations.py         CSV rows -> clean boxes per image (the one place parsing happens)
  split.py               balanced train/val/test split, patients kept together
  export_yolo.py         YOLO folders + dataset.yaml
  checks.py              data checks
  visualize.py, stats.py drawing boxes, dataset charts
  pipeline.py            runs the whole preparation
scripts/
  prepare_data.py        1. build the split and the YOLO dataset
  check_data.py          2. verify it
  visualize.py           3. look at it (replaces the old box plotter)
  data_stats.py          4. class counts, split balance, box sizes
  train.py               5. train YOLOv8
tests/                   pytest tests on a small fake dataset
```

Generated folders (not in git): `data/processed/`, `outputs/`, `runs/`.

## Setup on the server

Connect to the university VPN, then open the server in VS Code (Remote-SSH, bottom-left
icon, "Connect to Host"). In a terminal on the server:

```bash
cd /mnt/hdd2/faraj
git clone <your repo> dr-lesion-detection   # or copy the folder over with rsync
cd dr-lesion-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # should print True
nvidia-smi                                                    # pick a free GPU; the server is shared
```

Run long jobs inside `tmux` so they survive a dropped connection (`tmux new -s train`,
detach with Ctrl+b then d, come back with `tmux attach -t train`).

## Point it at the data

Edit `paths` in `configs/default.yaml`, or pass them on the command line. For example:

```yaml
paths:
  annotations: /mnt/hdd2/faraj/BBoxAnnotations
  images: /mnt/hdd2/faraj/train
  grades_csv: /mnt/hdd2/faraj/trainLabels.csv   # Kaggle image-level grades, or null
  output: data/processed
```

If your CSV column names differ from the defaults (`image_name`, `left_coordinate`, ...),
change `csv.columns` in the config.

## Workflow

```bash
# 1. Build the split and the YOLO dataset
python scripts/prepare_data.py

# 2. Check it (exits with an error if anything is wrong)
python scripts/check_data.py

# 3. Look at it: original CSV boxes, then the exported labels read back
python scripts/visualize.py --source csv
python scripts/visualize.py --source yolo --split train --zoom

# 4. Dataset charts and tables -> outputs/stats/
python scripts/data_stats.py

# 5. Sanity check: can YOLOv8 memorise 10 images? (mAP should get close to 1)
python scripts/train.py --overfit 10 --epochs 100

# 6. First real run
python scripts/train.py --size n --imgsz 1024 --epochs 50
```

Add `--dry-run` to any training command to see the settings without training. If the GPU
runs out of memory, lower `--batch` (or `--imgsz`). Use `--device 1` for another GPU.

Each run is saved in `runs/yolov8/<name>/` (weights, plots, `results.csv`, `run_info.json`
with settings and versions), and a summary line goes into `runs/experiments.csv`.

### What the checks cover

`check_data.py` re-reads the CSVs and confirms that every rectangle was either exported or
dropped for a logged reason, every label line is valid, every label converts back to the
same pixel box as the CSV (within 1 px), images and labels pair up, symlinks aren't broken,
and no image or patient is in two splits. It also warns if a lesion type is missing from
val or test.

### Looking at predictions

```bash
yolo detect predict model=runs/yolov8/<name>/weights/best.pt source=data/processed/yolo/images/val \
  imgsz=1024 save_txt=True save_conf=True
python scripts/visualize.py --source yolo --split val --predictions runs/detect/predict/labels --zoom
```

Annotations are solid, predictions dashed.

### Evaluating on the test set (once, at the end)

Training only uses train and val. When a model is final, evaluate the test split once:

```bash
yolo detect val model=runs/yolov8/<name>/weights/best.pt data=data/processed/yolo/dataset.yaml split=test imgsz=1024
```

## Decisions built into the code (be ready to explain them)

- **Split 70/15/15.** Agreed with the supervisor: with about 3,400 images, 10% would leave too few to test on.
- **Patients kept together.** A patient's left and right eyes go in the same split, because
  the two eyes look alike and splitting them would leak information into the test set.
- **Balanced split.** Iterative stratification (Sechidis et al., 2011) applied to patients, so
  each split gets about its share of images, left and right eyes, boxes of every lesion type
  and images of every DR grade. See `drdet/split.py` and check `data/processed/split_report.md`.
- **The split is fixed.** `splits.csv` is reused on every run, so later models will see the
  same data as YOLOv8. `--resplit` makes a new one; only do that on purpose, since old
  results stop being comparable.
- **Cleaning.** Coordinates are divided by the annotation tool's scale (as in the original
  plotter), clipped to the image, and boxes thinner than 1 px, outside the image, with
  unknown labels or duplicated are dropped. Every dropped box is counted in
  `data/processed/prepare_report.json`. Images with no boxes are left out (config option).
- **Class order** in the config sets the YOLO class ids. Don't change it mid-project.
- **Image size 1024** by default, because lesions are tiny. `outputs/stats/stats.md` shows what
  share of each lesion type would be under 8 px at 640, 1024 and 1280; use it to justify the choice.

## Roadmap

1. **YOLOv8 (now).** Get the data through YOLOv8, check the predictions look sensible, and
   settle image size and training settings.
2. **More YOLO models.** Add entries to `MODELS` in `scripts/train.py`:
   - YOLO26 (NMS-free): uses the same YOLO data; needs a recent `ultralytics`.
   - YOLOv5 (anchor-based): use the original repository (github.com/ultralytics/yolov5),
     because the `ultralytics` package's YOLOv5 ("YOLOv5u") is anchor-free. It reads the
     same `dataset.yaml`.
3. **COCO format, Faster R-CNN and DETR.** Add a COCO export next to `export_yolo.py`
   (driven by the same records and split), a PyTorch dataset over it, and training for
   Faster R-CNN (two-stage) and DETR (ask the supervisor which variant).
4. **Same evaluation for all.** Score every model with the same metric code on the same
   test split.

## Tests

```bash
python -m pytest -q
```

The tests build a small fake dataset with known edge cases (duplicates, scaled coordinates,
boxes off the edge, unknown labels, a missing image) and check parsing, the split, the
YOLO export and the checks.

## Troubleshooting

- **"image files reachable" fails:** the YOLO folders use symlinks to the original images.
  If you copied `data/processed` to another machine, or moved the images, run
  `prepare_data.py` again where the images are (or set `export.link_mode: copy`).
- **"images aren't in the saved split":** the annotations changed. Rerun with `--resplit`
  and retrain, since old runs used a different split.
- **Labels look stale in training:** `prepare_data.py` rebuilds the YOLO folder, which also
  deletes Ultralytics' `*.cache` files, so rerunning it fixes this.
