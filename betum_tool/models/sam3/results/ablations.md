# SAM 3 Ablation Log

Track all experiments here. One row per run.

## Results

| Run | Mode | Dataset | Prompts / Config | Confidence Threshold | AP@50 | Notes |
|-----|------|---------|-----------------|----------------------|-------|-------|
| 1   | zero-shot PCS | Cashew (val) | baseline prompts | 0.05 | — | — |
| 2   | zero-shot PCS | Coffee (val) | baseline prompts | 0.05 | — | — |
| 3   | zero-shot PCS | Cashew (val) | simple prompts | 0.05 | — | — |
| 4   | zero-shot PCS | Coffee (val) | simple prompts | 0.05 | — | — |

## Prompt Engineering Notes

Document which text prompts you tried and their effect on Promptable Concept
Segmentation quality and bounding box derivation.

| Prompt Set | Description | Effect on AP@50 |
|-----------|-------------|----------------|
| baseline | `"cashew tree"`, `"cashew flower"`, `"premature cashew nut"`, ... | — |
| simple | `"tree"`, `"flower"`, `"premature"`, ... | — |
| domain-specific | `"cashew tree foliage"`, `"cashew inflorescence"`, `"immature cashew nut"`, ... | — |

## Post-processing / Inference Notes

Document any tuning of confidence threshold, NMS, or mask-to-bbox conversion
settings.

| Parameter | Default | Tried | Effect |
|-----------|---------|-------|--------|
| confidence_threshold | 0.05 | — | — |
| min_mask_area_pixels | — | — | — |
| nms_iou_threshold | — | — | — |
