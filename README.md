# Diabetic Retinopathy Project

This repository prepares retinal-fundus lesion annotations into a dataset that can be inspected, validated, and exported to a YOLO-style object-detection format. The project is intentionally designed so that the large image corpus stays on the mounted storage drive and the repository only stores code, small metadata, and the annotation CSVs.

## 1. Project layout

The working data layout is:

- Repository code: `/home/faraj/Diabetic-Retinopathy-Project`
- Annotation CSV files: `/home/faraj/Diabetic-Retinopathy-Project/BBoxAnnotations`
- Training images: `/mnt/hdd2/faraj/train`
- Optional test images: `/mnt/hdd2/faraj/test`
- Generated dataset outputs: `/home/faraj/Diabetic-Retinopathy-Project/outputs`
- Python virtual environment: `/home/faraj/Diabetic-Retinopathy-Project/.venv`

Important: only the large image folders remain on the mounted drive. The pipeline reads directly from `/mnt/hdd2/faraj/train` and writes generated artifacts under the repository’s local `outputs/` folder instead of storing anything else on the mounted volume.

## 2. Repository structure

- `dr_data.py`: shared annotation parsing, box validation, and coordinate conversion
- `visualize_annotations.py`: plots boxes on a small sample of fundus images
- `yolo_loader.py`: builds train/val/test YOLO exports and audits annotation consistency
- `tests/test_yolo_data.py`: verifies split logic, scaling, validation rules, and patient safety
- `BBoxAnnotations/`: CSV annotation batches used to construct ground-truth lesion boxes

## 3. Intended workflow

### Step A — create the local environment

```bash
cd /home/faraj/Diabetic-Retinopathy-Project
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This keeps the dependency environment under the project folder, which is acceptable for this server while keeping the actual dataset files on the mounted HDD.

### Step B — visually inspect annotation alignment

The visual QA script overlays lesion boxes onto a few sample retinal images. This is meant to catch coordinate issues before training begins.

```bash
cd /home/faraj/Diabetic-Retinopathy-Project
. .venv/bin/activate
python visualize_annotations.py \
  --annotations /home/faraj/Diabetic-Retinopathy-Project/BBoxAnnotations \
  --images /mnt/hdd2/faraj/train \
  --output /home/faraj/Diabetic-Retinopathy-Project/outputs/visualizations/fundus_box_overlays.png \
  --max-images 6
```

What it does:

- loads CSV rows from the annotation folder
- keeps only rectangle annotations
- converts stored CSV coordinates back to pixel-space boxes using `image_scale_x` / `image_scale_y`
- clamps boxes to image boundaries
- rejects malformed, non-positive, and out-of-range lesions
- saves a small overlay image into the repository-local `outputs/` folder

### Step C — export the dataset to YOLO format

The exporter creates a YOLO dataset with normalized labels, image symlinks, train/val/test folders, class mapping, and a split audit file.

```bash
cd /home/faraj/Diabetic-Retinopathy-Project
. .venv/bin/activate
python yolo_loader.py \
  --annotations /home/faraj/Diabetic-Retinopathy-Project/BBoxAnnotations \
  --train-images /mnt/hdd2/faraj/train \
  --test-images /mnt/hdd2/faraj/test \
  --output /home/faraj/Diabetic-Retinopathy-Project/outputs/yolo_dataset \
  --seed 42
```

If `/mnt/hdd2/faraj/test` is absent, the script does not fail silently; it splits the annotated training set internally into 70% train, 15% validation, and 15% test. The split is patient-safe and class-aware, so left/right eyes and lesion classes are balanced as much as possible. If a real test folder exists, the importer keeps that folder separate and does not merge it into the training split.

### Step D — review the outputs

The export writes the following artifacts under the repository-local `outputs/` folder:

- `images/train/` and `images/val/` (symlinks to the original mounted image files)
- `images/test/` if a valid test folder is present
- `labels/train/`, `labels/val/`, `labels/test/`
- `dataset.yaml`
- `class_map.json`
- `train_val_splits.csv`

The script also prints an audit summary that includes:

- number of raw rows and duplicate rows filtered
- broken or malformed boxes dropped
- missing/corrupt image files
- box reconciliation counts
- patient count and split counts
- class counts by split

## 4. Data validation rules

The pipeline intentionally forces strict validation:

- rectangle rows only
- valid numeric coordinates
- positive width and height
- positive scale values
- boxes must be inside image bounds after scaling
- duplicate CSV rows are removed
- patient IDs are grouped so both eyes of the same patient stay in the same split
- train and test folders cannot share patient IDs

This is important because leakage across patients can make validation metrics look artificially strong.

## 5. Patient-safe split logic

The code groups images by patient ID and, when no separate test folder exists, it internally assigns patients to a 70/15/15 split: train/validation/test. A simple rule is used:

- `123_left.jpeg` and `123_right.jpeg` belong to the same patient
- any patient appearing in both the source training set and the source test set is rejected
- the training set is divided by patient into train, validation, and test subsets
- when an external test folder is present, it remains separate and is not re-partitioned into the internal training split

This design is intended to avoid optimistic validation results caused by patient-level data leakage while keeping the test set large enough to be meaningful.

## 6. Storage policy

The training and test images are large and can consume significant storage. The project therefore avoids copying them into the home directory by:

- reading directly from `/mnt/hdd2/faraj/train`
- reading directly from `/mnt/hdd2/faraj/test` when available
- writing generated dataset artifacts into `/home/faraj/Diabetic-Retinopathy-Project/outputs`
- creating symlinks rather than duplicates inside the export data structure

This preserves the home volume for code, metadata, and the virtual environment while keeping only the actual image corpus on the mounted disk. No generated model or dataset outputs should be stored under `/mnt/hdd2/faraj`.

## 7. Training and modeling direction

The current project is positioned as a foundation for lesion detection and image-level diabetic retinopathy grading.

### Detection baseline

- Use a PyTorch object-detection baseline such as Faster R-CNN with an FPN backbone
- Compare against a YOLO-based detector on the same train/val/test split
- Report per-class precision/recall and mAP at stated IoU thresholds

### Severity-grading baseline

- Train a model to predict the five ordered Kaggle DR severity grades
- Evaluate with quadratic weighted kappa and a confusion matrix

The system should not be presented as clinically validated from this dataset alone; rather, it is a reproducible research pipeline for building and comparing baselines.

## 8. GPU usage policy

The workstation has four A5000 GPUs. The supervisor explicitly indicated that the free cards for this project are GPUs 2 and 3, and these are the only ones that should be used for model training and inference.

Always launch PyTorch code with the visible subset restricted to those cards:

```bash
cd /home/faraj/Diabetic-Retinopathy-Project
. .venv/bin/activate
./run_on_gpus_2_3.sh python your_training_script.py
```

This sets `CUDA_VISIBLE_DEVICES=2,3`, which hides GPUs 0 and 1 from Python. Inside the process, torch sees only two devices, indexed as `cuda:0` and `cuda:1`. This means:

- `cuda:0` maps to physical GPU 2
- `cuda:1` maps to physical GPU 3
- do not use `device_ids=[2,3]` inside the training code after the environment variable is set
- use `device_ids=[0,1]` for a two-GPU run, or `torch.cuda.set_device(0)` for a single-GPU run

For future multi-GPU training, the workflow should therefore be:

```bash
cd /home/faraj/Diabetic-Retinopathy-Project
. .venv/bin/activate
./run_on_gpus_2_3.sh python -m torch.distributed.run --nproc_per_node=2 train.py
```

This prevents accidental use of GPUs 0 and 1, which are already reserved by other users.

## 9. Operational cautions

- Do not run a dataset export with a training folder that has been copied into the home directory.
- Prefer absolute paths to the mounted drive for any large image data.
- Keep the virtual environment under the repo (for example `.venv`) to satisfy local environment requirements.
- Use only the allowed GPUs for model runs: physical GPUs 2 and 3.
- If the annotation CSV folder changes location, update the `--annotations` path in the commands above.
- If a test folder is not yet available, the exporter creates an internal 70/15/15 train/validation/test split from the annotated training images.

## 10. Validation status

The repository includes automated tests that check:

- box scaling and clamping
- rejection of malformed or invalid boxes
- deterministic train/val split assignment
- patient grouping across both eyes
- class distribution across splits
- official test-folder handling

The project tests are passing under the current environment.

## 11. Example end-to-end command sequence

```bash
cd /home/faraj/Diabetic-Retinopathy-Project
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python visualize_annotations.py \
  --annotations /home/faraj/Diabetic-Retinopathy-Project/BBoxAnnotations \
  --images /mnt/hdd2/faraj/train \
  --output /home/faraj/Diabetic-Retinopathy-Project/outputs/visualizations/fundus_box_overlays.png \
  --max-images 6

python yolo_loader.py \
  --annotations /home/faraj/Diabetic-Retinopathy-Project/BBoxAnnotations \
  --train-images /mnt/hdd2/faraj/train \
  --test-images /mnt/hdd2/faraj/test \
  --output /home/faraj/Diabetic-Retinopathy-Project/outputs/yolo_dataset \
  --seed 42
```

This is the intended operating pattern for the project: keep the repo and environment local, keep the image corpus on the HDD, and store generated datasets and training artifacts in the repository’s local `outputs/` folder only.

## 12. Research reading list

1. Gulshan et al. (2016), Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs, JAMA.
2. Ting et al. (2017), Development and Validation of a Deep Learning System for Diabetic Retinopathy and Related Eye Diseases Using Retinal Images from Multiethnic Populations, JAMA.
3. Abràmoff et al. (2018), Pivotal trial of an autonomous AI-based diagnostic system for detection of diabetic retinopathy in primary care offices, npj Digital Medicine.
4. Redmon et al. (2016), You Only Look Once: Unified, Real-Time Object Detection, CVPR.
5. Ren et al. (2015), Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, NeurIPS.
6. Lin et al. (2017), Feature Pyramid Networks for Object Detection, CVPR.
7. Lin et al. (2017), Focal Loss for Dense Object Detection, ICCV.
8. Tan et al. (2020), EfficientDet: Scalable and Efficient Object Detection, CVPR.

The project is designed to move from raw annotation parsing to a clean, audit-friendly, patient-safe detection pipeline before model training begins.