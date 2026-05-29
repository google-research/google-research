# YOLO26 Ablation Log

Track all supervised training experiments here. One row per run.

## Results

| Run | Dataset | Model Size | Epochs | Batch Size | Img Size | Augmentations | AP@50 | Notes |
|-----|---------|------------|--------|------------|----------|---------------|-------|-------|
| 1   | Cashew  | yolo26n    | 50     | 16         | 640      | Default       | —     | Baseline run |
| 2   | Coffee  | yolo26n    | 50     | 16         | 640      | Default       | —     | Baseline run |

## Hyperparameter Notes

Document any tuning of training hyperparameters.

| Parameter | Default | Tried | Effect |
|-----------|---------|-------|--------|
| learning_rate | 0.01 | — | — |
| epochs | 50 | — | — |
| batch_size | 16 | — | — |
| imgsz | 640 | — | — |
