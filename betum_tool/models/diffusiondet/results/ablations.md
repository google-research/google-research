# DiffusionDet Ablation Log

Track all experiments here. One row per run.

## Results

| Run | Mode | Dataset | Config (Backbone, Iterations) | Denoising Steps | Proposal Boxes | AP@50 | Notes |
|-----|------|---------|--------------------------------|-----------------|----------------|-------|-------|
| 1 | fine-tuned | Cashew (val) | ResNet-50, 5000 iter, bs=2 | 4 | 500 | — | Baseline |
| 2 | fine-tuned | Coffee (val) | ResNet-50, 5000 iter, bs=2 | 4 | 500 | — | Baseline |
| 3 | fine-tuned | Cashew (val) | ResNet-50, 5000 iter, bs=2 | 8 | 500 | — | Increasing denoising steps |
| 4 | fine-tuned | Coffee (val) | ResNet-50, 5000 iter, bs=2 | 8 | 500 | — | Increasing denoising steps |

## Denoising Steps Study

Document how the number of inference denoising steps (`num_inference_steps` or
similar) affects the trade-off between detection accuracy (AP@50) and inference
speed.

| Denoising Steps | Cashew AP@50 | Coffee AP@50 | Inference Speed (sec/img) |
|-----------------|--------------|--------------|--------------------------|
| 1 | — | — | — |
| 2 | — | — | — |
| 4 | — | — | — |
| 8 | — | — | — |

## Hyperparameter & Config Notes

Document any tuning of fine-tuning hyperparameters or architecture decisions.

| Parameter | Default | Tried | Effect |
|-----------|---------|-------|--------|
| backbone | ResNet-50 | — | — |
| num_proposals | 500 | — | — |
| learning_rate | 1e-4 | — | — |
| max_iter | 5000 | — | — |
| batch_size | 2 | — | — |
