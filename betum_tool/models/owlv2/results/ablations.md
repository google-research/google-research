# OWL-v2 Ablation Log

Track all experiments here. One row per run.

## Results

| Run | Mode | Dataset | Prompts / Config | Threshold | AP@50 | Notes |
|-----|------|---------|-----------------|-----------|-------|-------|
| 1 | zero-shot | Cashew (val) | baseline prompts | 0.3 | — | — |
| 2 | zero-shot | Coffee (val) | baseline prompts | 0.3 | — | — |
| 3 | fine-tuned | Cashew (val) | 1 epoch, lr=1e-5, bs=4 | 0.3 | — | — |
| 4 | fine-tuned | Coffee (val) | 1 epoch, lr=1e-5, bs=4 | 0.3 | — | — |

## Prompt Engineering Notes

Document which text prompts you tried and their effect on detection quality.

| Prompt Set | Description | Effect on AP@50 |
|-----------|-------------|----------------|
| baseline | `"unripe coffee cherry"`, `"ripe coffee cherry"`, … | — |

## Hyperparameter Notes

Document any tuning of fine-tuning hyperparameters.

| Parameter | Default | Tried | Effect |
|-----------|---------|-------|--------|
| learning_rate | 1e-5 | — | — |
| num_epochs | 1 | — | — |
| batch_size | 4 | — | — |
| confidence_threshold | 0.3 | — | — |
