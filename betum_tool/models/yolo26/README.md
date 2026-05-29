# Group 2 — YOLO26: Real-Time Supervised Detection

## Overview

This module implements **YOLO26** for real-time, supervised object detection on
the Coffee & Cashew dataset.

**Model:** [YOLO26](https://huggingface.co/Ultralytics/YOLO26)

**Task:** Train and optimize a fast, task-specific detection model directly on
the provided plant data. YOLO architectures are built for real-time efficiency —
expect rapid training cycles and spend the majority of time tuning
hyperparameters.

## Directory Structure

```
yolo26/
├── README.md                    # This file
├── notebooks/
│   └── train_and_infer.ipynb    # Training + inference Colab notebook
├── scripts/
│   └── parse_output.py          # Parse YOLO outputs → COCO JSON
└── results/
    └── ablations.md             # Log AP@50 per experiment
```

## Quick Start

1. Open `notebooks/train_and_infer.ipynb` in Google Colab
2. Follow the cells to download data, train the model, and run inference
3. Log your results in `results/ablations.md`

## Evaluation

All predictions must be converted to COCO JSON format for unified evaluation:

```python
from shared.evaluate import evaluate_coco
results = evaluate_coco(gt_path="path/to/gt.json", pred_path="path/to/predictions.json")
```

## Key Hyperparameters to Explore

- Image size (640, 1280)
- Batch size
- Learning rate & scheduler
- Number of epochs
- Augmentation strategies
- Confidence & NMS thresholds
