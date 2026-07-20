# Bëtum Tool

**Bëtum Tool** (derived from the Wolof phrase for **"The Eye of the Field"**) is
a unified benchmarking toolkit built for the **Artemis Workshop (Dakar, June
1–5)**.

This toolkit represents the image processing and visual data extraction
components of the workshop. It focuses on applied Computer Vision methods,
enabling students to program systems that can "see" and accurately measure
phenotypic traits from plant imagery. Using **Bëtum Tool**, students will run,
evaluate, and compare state-of-the-art object detection profiles on local
agricultural datasets.

## Goal

Four student teams each tackle the same detection task with a different
architecture, then compare results using a single metric: **AP@50** via
COCOEval.

| Group | Model                          | Approach                            |
| ----- | ------------------------------ | ----------------------------------- |
| 1     | OWL-ViT                        | Open-vocabulary zero-shot detection |
| 2     | YOLO26                         | Real-time supervised detection      |
| 3     | SAM 3                          | Promptable segmentation → bounding  |
:       :                                : boxes                               :
| 4     | DiffusionDet / InstructPix2Pix | Generative perception               |

For baseline experimental findings, architectural benchmarks, and data-level
limitations identified during previous workshops, see the
[Technical Briefing](TECHNICAL_BRIEFING.md).

## Dataset

**Coffee & Cashew Nut**
([Sanya et al., 2024](https://doi.org/10.1016/j.dib.2023.109952)) - a yield
estimation / fruit maturity dataset collected via drone in Uganda. Images are
annotated in YOLO format; we convert to COCO JSON for evaluation.

**Coffee** - 3,000 images · 126,840 annotations · 5 classes

ID  | Class         | Count   | %
--- | ------------- | ------: | ---:
0   | `unripe`      | 121,761 | 96.0
1   | `ripening`    | 630     | 0.5
2   | `ripe`        | 1,188   | 0.9
3   | `spoilt`      | 201     | 0.2
4   | `coffee_tree` | 3,060   | 2.4

**Cashew** - 3,086 images · 88,364 annotations · 6 classes

ID  | Class       | Count  | %
--- | ----------- | -----: | ---:
0   | `tree`      | 5,347  | 6.1
1   | `flower`    | 23,169 | 26.2
2   | `premature` | 21,200 | 24.0
3   | `unripe`    | 5,347  | 6.1
4   | `ripe`      | 7,481  | 8.5
5   | `spoilt`    | 25,820 | 29.2

## Dataset & Attribution

This project utilizes the **Coffee & Cashew Nut** dataset published by Sanya et
al. (2024), which is hosted on Mendeley Data and licensed under the **Creative
Commons Attribution 4.0 International (CC BY 4.0)** license.

*   **Original Dataset & Paper:** Sanya, et al. (2024). *A dataset of coffee and
    cashew nut maturity stages and yield estimation in Uganda*. Data in Brief.
    [DOI Link](https://doi.org/10.1016/j.dib.2023.109952) |
    [Mendeley Page](https://data.mendeley.com/datasets/r46c6bpfpf/1)

*   **Modifications:** Original bounding box annotations in YOLO format have
    been converted to COCO JSON splits (e.g., `cashew_val.json`,
    `coffee_val.json`) using `common/yolo_to_coco.py` to enable standardized
    validation via COCOEval.

## Repository Structure

```
Betum_Tool/
├── README.md
├── requirements.txt
│
├── common/                        # Cross-group utilities
│   ├── yolo_to_coco.py            #   YOLO → COCO JSON converter
│   ├── class_map.json             #   Dataset label ↔ category mapping
│   ├── evaluate.py                #   Unified COCOEval AP@50 wrapper
│   └── visualize.py               #   Prediction visualisation
│
└── models/
    ├── owlv2/                     # Group 1
    │   └── notebooks/inference.ipynb
```

Each group works only in their own `models/<name>/` directory. Results
(hyperparameters, prompts, AP@50 scores) are logged in
`models/<name>/results/ablations.md` on each team's fork.

## Usage

1.  **Fork** this repository.
2.  Open your group's Colab notebook in Google Colab (links in each
    `models/<name>/README.md`).
3.  The notebook handles environment setup, data download, and model
    inference/training.
4.  Record your experiments in `results/ablations.md`.
5.  All final predictions must be formatted as COCO JSON and evaluated with
    `common/evaluate.py`.

## Running Scripts

From the parent directory (`google-research/`), you can run python scripts in
this directory using:

```bash
python -m betum_tool.path.to.script
```

For example, to run a test script or evaluate:

```bash
python -m betum_tool.common.evaluate --prediction_file path/to/predictions.json
```

## License

Apache 2.0 — see
[LICENSE](https://github.com/google-research/google-research/blob/master/LICENSE).

--------------------------------------------------------------------------------

This is not an officially supported Google product. This project is not eligible
for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).
