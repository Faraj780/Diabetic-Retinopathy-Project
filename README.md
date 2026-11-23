# Diabetic retinopathy lesion detection

Benchmark of object detectors on Kaggle DR fundus images with bounding-box lesion annotations
(9 lesion classes, see `configs/default.yaml`). The aim, as agreed with the supervisor, is a
working benchmark that compares detector families on the same data, not a target accuracy.

**Current stage: YOLOv8 only.** Other formats and models come later (see [Roadmap](#roadmap)).

## Layout

```
configs/default.yaml   paths, classes, label merging, split ratios, allowed GPUs
drdet/
  annotations.py       CSV rows -> boxes per image (the one place parsing happens)
  split.py             train/val/test split, patients kept together
  export_yolo.py       YOLO folders + dataset.yaml
  pipeline.py          runs the three steps above
scripts/
  prepare_data.py      1. build the split and the YOLO dataset
  train.py             2. train YOLOv8
tests/                 pytest tests on a small fake dataset
```

Generated (not in git): `data/processed/` (split + YOLO dataset), `runs/` (training runs),
`outputs/` (`sample_model_input.jpg`: one training image exactly as YOLOv8 receives it).

## Setup on the server

The project, annotation CSVs and venv live in the home folder. `/mnt/hdd2/faraj/train` holds
only the Kaggle training images and is read-only: nothing is written there (the YOLO dataset
uses symlinks to the images).

```bash
cd /home/faraj/Diabetic-Retinopathy-Project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # should print True
nvidia-smi                                                    # shared server: we may only use GPUs 2 and 3
```

Run long jobs inside `tmux` so they survive a dropped connection (`tmux new -s train`, detach
with Ctrl+b then d, come back with `tmux attach -t train`).

## Workflow

```bash
python scripts/prepare_data.py                              # 1. split + YOLO dataset
python scripts/train.py --epochs 1                          # 2. quick check that training runs
python scripts/train.py --size n --imgsz 1024 --epochs 50   # 3. first real run
```

If a GPU runs out of memory, lower `--batch` (or `--imgsz`). Other Ultralytics settings:
`--set lr0=0.001`. Resume an interrupted run with `--resume runs/yolov8/<name>/weights/last.pt`.
Each run is saved in `runs/yolov8/<name>/` (weights, `results.csv`, plots, and
`train_batch*.jpg` showing augmented training batches).

**GPUs:** only GPUs 2 and 3 (`training.allowed_gpus`). `train.py` hides the others from
PyTorch and uses both by default (`--batch` is the total over both); `--device 2` or
`--device 3` uses one. Commands that don't go through `train.py` (like `yolo ...` below) need
`CUDA_VISIBLE_DEVICES=2,3` in front, otherwise they default to GPU 0.

### Evaluating on the test set (once, at the end)

Training only uses train and val. When a model is final, evaluate the test split once:

```bash
CUDA_VISIBLE_DEVICES=2,3 yolo detect val model=runs/yolov8/<name>/weights/best.pt \
  data=data/processed/yolo/dataset.yaml split=test imgsz=1024
```

## Decisions built into the code (be ready to explain them)

- **Split 70/15/15**, agreed with the supervisor: with about 3,400 images, 10% would leave too few to test on.
- **Patients kept together.** A patient's left and right eyes go in the same split, because
  the two eyes look alike and splitting them would leak information into the test set.
- **Balanced split.** Iterative stratification (Sechidis et al., 2011) over patients, so each
  split gets about its share of images, boxes of every class and images of every DR grade.
- **The split is fixed.** `data/processed/splits.csv` is reused on every run so later models
  see the same data. `--resplit` makes a new one; only do that on purpose.
- **DR grades** come from the `whole_image` rows in the CSVs (Kaggle's `trainLabels.csv` isn't
  on the server) and are only used to balance the split. Graded twice differently: higher grade.
- **Haemorrhages merged into one class.** Annotators used the generic "Retinal Haemorrhages"
  for about 5,200 boxes, often in the same images as "Dot" and "Blot" boxes, so dot vs blot
  isn't labelled consistently. Dropping the generic boxes would leave real haemorrhages
  unlabelled (the model would learn they're background), so all three map to "Retinal
  Haemorrhage" (`label_map` in the config). Pre-retinal haemorrhage, IRMA and venous beading
  are kept as classes for the same reason; retinal thickening (2 boxes) and vitreous
  haemorrhage (1) are dropped.
- **Cleaning.** Exact duplicate CSV rows are removed (the two annotation folders overlap).
  Coordinates are divided by the annotation tool's scale, clipped to the image; boxes under
  1 px and duplicate boxes are dropped. Images without boxes are left out.
- **Image size 1024** by default, because lesions are tiny: at 640 many microaneurysms would
  be only a few pixels wide.
- **Class order** in the config sets the YOLO class ids. Don't change it mid-project.

## Roadmap

1. **YOLOv8 (now).** Get the data through YOLOv8 and settle image size and training settings.
2. **More YOLO models.** YOLO26 (same YOLO data, recent `ultralytics`); YOLOv5 anchor-based
   from the original repository (github.com/ultralytics/yolov5), which reads the same `dataset.yaml`.
3. **COCO format, Faster R-CNN and DETR.** A COCO export next to `export_yolo.py`, driven by
   the same records and split.
4. **Same evaluation for all.** Every model scored with the same metric code on the same test split.

## Tests

```bash
python -m pytest -q
```
