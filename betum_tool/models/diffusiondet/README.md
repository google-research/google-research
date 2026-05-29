# Group 4 — DiffusionDet / InstructPix2Pix: Generative Perception

## Overview

This module implements a **generative approach to object detection**, using
**DiffusionDet** as the underlying model.

### Option A: DiffusionDet

**Model:** [DiffusionDet](https://github.com/ShoufaChen/DiffusionDet)

**Task:** Formulate object detection as a denoising process — generate bounding boxes from noisy boxes, proving that generative models can be repurposed for strict vision tasks.

### Option B: InstructPix2Pix (Alternative)

**Model:** [InstructPix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix)

**Task:** Repurpose a generative image editor as a localisation tool by giving
it text instructions to highlight regions, computing pixel-level differences
between original and edited images, and extracting bounding boxes from the
difference maps.

## Directory Structure

```
diffusiondet/
├── README.md                    # This file
├── notebooks/
│   └── train_and_infer.ipynb    # Training + inference Colab notebook
├── scripts/
│   ├── register_dataset.py      # Register Coffee/Cashew with Detectron2
│   └── parse_output.py          # Parse model outputs → COCO JSON
└── results/
    └── ablations.md             # Log AP@50 per experiment
```

## Quick Start

1. Open `notebooks/train_and_infer.ipynb` in Google Colab
2. Follow the cells to set up the environment, register the dataset, and train/infer
3. Log your results in `results/ablations.md`

## Evaluation

All predictions must be converted to COCO JSON format for unified evaluation:

```python
from shared.evaluate import evaluate_coco
results = evaluate_coco(gt_path="path/to/gt.json", pred_path="path/to/predictions.json")
```

## Key Parameters to Explore

### DiffusionDet

- Number of diffusion steps
- Number of proposal boxes
- SNR scale
- Training iterations

### InstructPix2Pix

- Text prompt engineering ("highlight all ripe coffee cherries")
- Image guidance scale
- Difference threshold for bbox extraction
- Post-processing morphological operations
