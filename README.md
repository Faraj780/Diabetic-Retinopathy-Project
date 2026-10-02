# Diabetic-Retinopathy-Project

## Annotation inspection

The repository contains batched annotation CSVs under `BBoxAnnotations/` and a local `train/` image folder. The image folder and archive are excluded from Git because of their size.

Create the environment and run the summary plot:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python visualize_annotations.py
```

The summary is written to `outputs/visualizations/annotation_summary.png`. Generate a small box-overlay grid from the training images with:

```sh
python visualize_annotations.py --images train --max-images 6
```

The script checks filenames directly under the image folder rather than scanning image contents. CSV box coordinates are divided by their recorded `image_scale_x` and `image_scale_y` values to map them to the original image dimensions. Identical rows repeated across the full and stage-specific exports are counted once in summaries.

## Initial data observations

- Annotation types include image-level DR grades and lesion bounding boxes, so this is a multi-task dataset rather than box detection alone.
- Lesion classes and severity grades are imbalanced, and most boxes are small. The generated summary plot provides an initial view of both.
- Keep patient/image grouping intact when making train/validation/test splits; images from the same patient should not leak across splits if patient identifiers can be recovered.
- Before training, validate coordinate mapping visually on several images and audit labels, duplicate annotations, missing images, and boxes outside image bounds.

## Initial modeling direction

Use **PyTorch** for the first benchmark because its `torchvision` detection models provide a straightforward Faster R-CNN implementation and make a reproducible two-stage baseline practical. Start with a pretrained **Faster R-CNN with an FPN backbone** for lesion detection, then compare against a current **YOLO** model using the same images, splits, and evaluation protocol. FPN is relevant because lesion sizes vary, while the observed small boxes mean image resolution and tile-based experiments may matter as much as detector choice. Avoid committing to a single final architecture until both baselines have been measured.

Treat image-level severity grading as a separate baseline initially: predict the five ordered Kaggle severity grades from the full image, using an ordinal-aware evaluation such as quadratic weighted kappa. A later unified system can combine grade predictions with lesion counts/types or multi-task features. Detection should be evaluated with per-class precision/recall and mAP at stated IoU thresholds; severity should use quadratic weighted kappa and a confusion matrix. Do not describe either as clinically validated from this dataset alone.

## Starter reading list

These eight papers give a focused first pass across retinal grading and the YOLO/R-CNN detector families:

1. Gulshan et al. (2016), [Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs](https://doi.org/10.1001/jama.2016.17216), *JAMA*.
2. Ting et al. (2017), [Development and Validation of a Deep Learning System for Diabetic Retinopathy and Related Eye Diseases Using Retinal Images From Multiethnic Populations](https://doi.org/10.1001/jama.2017.18152), *JAMA*.
3. Abràmoff et al. (2018), [Pivotal trial of an autonomous AI-based diagnostic system for detection of diabetic retinopathy in primary care offices](https://doi.org/10.1038/s41746-018-0040-6), *npj Digital Medicine*.
4. Redmon et al. (2016), [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640), CVPR.
5. Ren et al. (2015), [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497), NeurIPS.
6. Lin et al. (2017), [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144), CVPR.
7. Lin et al. (2017), [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002), ICCV.
8. Tan et al. (2020), [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070), CVPR.