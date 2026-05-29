# Group 1 — OWL-v2: Open-Vocabulary Object Detection

## Overview

OWL-v2 (`google/owlv2-base-patch16-ensemble`) is an **open-vocabulary** object
detector. Unlike traditional detectors that are trained on a fixed set of
classes, OWL-v2 accepts **text queries** at inference time: you describe what
to find, and it localises it.

This group has two objectives:

1. **Zero-shot inference**: detect Coffee/Cashew objects using text prompts
   alone (no training).
2. **Fine-tuning**: adapt OWL-v2 to the dataset using the repo's existing
   training pipeline, then compare AP@50 against the zero-shot baseline.

## Directory Layout

```
models/owlv2/
├── README.md              ← You are here
├── notebooks/
│   ├── inference.ipynb    # Zero-shot inference notebook
│   └── finetune.ipynb     # Fine-tuning notebook
├── scripts/
│   └── parse_output.py    # OWL-v2 output → COCO predictions JSON
└── results/
    └── ablations.md       # Log your experiments here
```

## Data Setup

> **Important:** Data is downloaded from the source in each Colab notebook; it
> never lives in the repo. The `data/` directory is git-ignored.

### Step 1 — Download from Mendeley

Download the Coffee & Cashew Nut dataset from [Mendeley
Data](https://data.mendeley.com/datasets/r46c6bpfpf/1) (Sanya et al., 2024). In
Colab, you can use `wget` or the Mendeley API. Locally, download and unzip into
`data/`:

```
data/
├── Cashew/
│   └── Cashew-Uganda/
│       ├── images/        # 3 086 images
│       └── Labels/        # YOLO .txt annotations (capital L)
└── Coffee/
    ├── Batch1/            # 10 batches, each with images/ + labels/
    │   ├── images/
    │   ├── labels/
    │   └── Side view/     # (some batches only, ignored)
    ├── Batch2/
    │   ├── images/
    │   └── labels/
    ├── ...
    └── Batch10/
```

### Step 2 — Flatten Coffee batches

The Coffee dataset is split across 10 batch directories. Merge them into a
single `images/` + `labels/` layout before conversion. Filenames are
batch-prefixed to avoid collisions; duplicate files (`Copy of DJI_…`) are
automatically skipped.

```bash
python common/flatten_coffee.py
# Reads from data/Coffee/Batch*  →  writes to data/Coffee_flattened/images + labels
```

### Step 3 — Convert YOLO → COCO JSON

Use `common/yolo_to_coco.py` to produce the COCO JSON files needed for
evaluation and fine-tuning:

```bash
# Cashew
python common/yolo_to_coco.py \
    --images data/Cashew/Cashew-Uganda/images \
    --labels data/Cashew/Cashew-Uganda/Labels \
    --class_map common/class_map.json \
    --dataset cashew \
    --output data/coco/ \
    --split_ratio 0.8

# Coffee (after flattening)
python common/yolo_to_coco.py \
    --images data/Coffee_flattened/images \
    --labels data/Coffee_flattened/labels \
    --class_map common/class_map.json \
    --dataset coffee \
    --output data/coco/ \
    --split_ratio 0.8
```

This produces `data/coco/cashew_train.json`, `cashew_val.json`,
`coffee_train.json`, `coffee_val.json`.

> See `common/class_map.json` for the canonical class-to-ID mapping for both
> datasets.

---

## Running the Notebooks

### Zero-Shot Inference (`notebooks/inference.ipynb`)

Open in Colab (or run locally). Set `NUM_EXAMPLES = 4` for a quick smoke test,
or remove the limit for a full run.

The notebook loads the model, runs inference using text prompts, evaluates
with COCOEval AP@50, and visualises predictions vs. ground truth.

**Requires:** Steps 1–3 above (COCO JSON for the image list and ground truth).

### Fine-Tuning (`notebooks/finetune.ipynb`)

Open in Colab (or run locally). Set `NUM_EXAMPLES = 4` for a quick smoke test
(1 epoch, batch=4).

The notebook converts the COCO JSON to a HuggingFace `DatasetDict` inline
(using the logic from `common/data.py`), then fine-tunes OWL-v2 on the training
split, evaluates on validation, and compares fine-tuned AP@50 against the
zero-shot baseline.

**Requires:** Steps 1–3 above (COCO JSON). The COCO → HuggingFace conversion
happens inside the notebook.

### Evaluation

Both notebooks call `common/evaluate.py` at the end to compute AP@50 via
COCOEval. Log your results in `results/ablations.md`.

---

## Key Concepts

- **Text prompts** replace class IDs at inference time. Prompt engineering
  matters: try `"ripe coffee cherry"` vs `"ripe cherry"` vs `"red fruit"`.
- **Fine-tuning** uses a DETR-style `SetCriterion` with Hungarian matching.
  The repo provides this out of the box in `common/losses.py` and
  `common/matcher.py`.
- **Boxes** are normalised to `[cx, cy, w, h]` relative to the longest image
  edge (OWL-v2's convention). See
  `models/owlv2/owl.py::normalize_annotation_for_owlv2()`.

## Dependencies

See the root `requirements.txt`. Key packages: `transformers>=4.40`, `torch`,
`torchmetrics`, `pycocotools`, `datasets`.
