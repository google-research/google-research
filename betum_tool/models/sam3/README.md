# Group 3 — SAM 3: Promptable Foundation Extraction

## Overview

This module implements **SAM 3** (Segment Anything Model 3) for dense instance
segmentation, with a pipeline to convert masks to bounding boxes for
unified AP@50 evaluation.

**Model:** [SAM 3](https://huggingface.co/facebook/sam3)

**Task:** Run heavy inference or fine-tune SAM 3's mask decoder to extract dense
instance segmentation masks. To align with the workshop object detection
goal, engineer a pipeline that converts pixel-perfect masks into standard
bounding boxes by extracting the minimum and maximum coordinates.

## Directory Structure

```
sam3/
├── README.md                    # This file
├── notebooks/
│   └── template_inference.ipynb          # Inference Colab notebook
├── scripts/
│   └── masks_to_bboxes.py       # Convert SAM masks → COCO bounding boxes
└── results/
    └── ablations.md             # Log AP@50 per experiment
```

## Quick Start

1. Open `notebooks/template_inference.ipynb` in Google Colab
2. Follow the cells to download data, run SAM 3 inference, and convert masks to boxes
3. Log your results in `results/ablations.md`

## Mask → BBox Pipeline

```python
# Extract bounding box from a binary mask
from scripts.masks_to_bboxes import mask_to_bbox
bbox = mask_to_bbox(binary_mask)  # Returns [x_min, y_min, width, height]
```

## Evaluation

All predictions must be converted to COCO JSON format for unified evaluation:

```python
from shared.evaluate import evaluate_coco
results = evaluate_coco(gt_path="path/to/gt.json", pred_path="path/to/predictions.json")
```

## Key Parameters to Explore

- Point prompts vs. box prompts vs. text prompts
- Number of points per object
- Mask decoder fine-tuning (frozen encoder vs. full fine-tuning)
- Post-processing: NMS on extracted bounding boxes
- Confidence thresholds
