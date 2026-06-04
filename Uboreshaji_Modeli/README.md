# Uboreshaji Modeli – Multi-Modal Fine-Tuning Framework

This is not an officially supported Google product. This project is not eligible
for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

**Uboreshaji Modeli** (Kiswahili for "model fine-tuning") is a modular PyTorch
framework for fine-tuning vision, vision-language, text, and speech foundation
models on specialized downstream tasks.

--------------------------------------------------------------------------------

## 🌟 What's New: Multi-Modal Architectural Extension

The framework has been significantly expanded beyond its original OWL-v2 object
detection roots to support a rich suite of multimodal foundation models. By
adopting the **Composable Strategy Pattern**, the framework achieves clean
separation of concerns and configuration-driven workflows.

### Intended Modalities & Workflows

*   **Vision VLM (Gemma 3 / 4)**: Detection and Instance Segmentation via
    location tokenization (`<loc0000>` to `<loc1023>`).

*   **Text SFT (Gemma 3 / 4)**: Text-to-text instructional fine-tuning (e.g.,
    Swahili translation) with chat templating and prompt masking (`-100`).

*   **Audio ASR (e.g. Gemma Audio, Whisper, MMS)**: Automatic Speech Recognition
    fine-tuning, featuring Universal Speech Model (USM) encoder integration and
    `Seq2SeqTrainer` support.

*   **Vision Detection (OWL-v2)**: DETR-style bounding box object detection with
    bipartite matching (`HungarianMatcher`) and `SetCriterion` focal loss.

--------------------------------------------------------------------------------

## 📁 Folder Structure

```
Uboreshaji_Modeli/
├── README.md                    # Complete project overview & index
├── main.py                      # CLI frontend wrapper (allocations, flags)
├── main_lib.py                  # Universal modality-agnostic orchestrator
├── requirements.txt             # Core Python dependencies
├── run.sh                       # Automated environment setup & verification
├── workflow.ipynb               # Interactive Jupyter notebook walkthrough
├── common/                      # Shared infrastructure & data utilities
│   ├── config.py                # Global configurations and enums
│   ├── config_utils.py          # Config loading and paths derivation
│   ├── data.py                  # HF datasets, streaming & DDP locks
│   ├── metrics.py               # Modal evaluation (WER, BLEU, mAP)
│   └── trainer.py               # Custom training loop extensions
├── engines/                     # Composed ModelEngine coordinators
│   ├── base.py                  # Composed ModelEngine & protocols ABCs
│   ├── decoders.py              # Bounding box, polygon, and CTC decoders
│   ├── factory.py               # Engine assembler resolver factory
│   └── [model-specific engines] # Composed engines (owl, gemma_vision)
└── trainers/                    # Concrete training strategies
    ├── base.py                  # TrainerStrategy interface ABC
    ├── detection.py             # OWL-v2 object detection strategy
    └── factory.py               # Task-to-strategy resolver factory
```

--------------------------------------------------------------------------------

## 🎯 Key Features

1.  **Modality-Agnostic Orchestration**: Single unified training flow
    (`main_lib.py`) that isolates execution into composed strategies.
2.  **Pluggable Engines (`ModelEngine`)**: Composes dynamic `DataPreprocessor`,
    `LossHandler` (optional), and `PredictionDecoder` (optional).
3.  **Pluggable Strategies (`TrainerStrategy`)**: Decouples trainers (Hugging
    Face `Trainer`, `CustomSFTTrainer`, and distributed DDP process group loops)
    from engines.
4.  **Borg distributed optimizations**:
    *   Lock-free local cache downloads for multi-GPU DDP training.
    *   Streaming bypass support for infinite iterable speech datasets.
    *   Early memory segmentation allocations to prevent CUDA out-of-memory
        errors.

--------------------------------------------------------------------------------

## 🚀 How to Get Started

### 1. Automated Quick Verification

```bash
cd google-research
./Uboreshaji_Modeli/run.sh
```

This automatically sets up a virtual environment, installs dependencies, and
runs all unit and integration tests.


For exhaustive configuration details across vision, text, and speech, see the
[USER_GUIDE.md](google/USER_GUIDE.md).

--------------------------------------------------------------------------------

## 🔗 License

Apache 2.0 (see file header for details)
