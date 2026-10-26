# Privacy Auditing for Synthetic Data

This repository contains code for privacy auditing of large language model fine-
tuning, corresponding to our research on empirical privacy measurement using
canary auditing.

## Overview

This codebase implements:

- **DP fine-tuning** of Gemma 3 models (1B and 12B) on the Enron email dataset
- **Canary-based privacy auditing** to measure empirical ε-DP guarantees
- **Synthetic data generation** and evaluation via MAUVE scores

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

You will also need to install:

- [tunix](https://github.com/google-deepmind/tunix) — fine-tuning framework
- [gemma](https://github.com/google-deepmind/gemma) — Gemma model weights and utilities
- [qwix](https://github.com/google/qwix) - LoRA implementation

### Configure dataset and checkpoint paths

Set the following environment variables:

```bash
# Path to the Enron training dataset (.bagz format)
export ENRON_TRAIN_PATH="/path/to/enron/train.bagz"

# Path to the Enron test dataset
export ENRON_TEST_PATH="/path/to/enron/test.bagz"

# Path to the Gemma 3 tokenizer model
export GEMMA3_TOKENIZER_PATH="/path/to/gemma3/tokenizer.model"
```

You can download the Enron email dataset from public sources (e.g. [Enron Email Dataset](https://www.cs.cmu.edu/~enron/)).
Gemma 3 model checkpoints are available on [Kaggle](https://www.kaggle.com/models/google/gemma-3).

## Usage

All scripts use `python -m`. Run from the parent directory of
`privacy_auditing/`.

### Training

```bash
# Full fine-tuning on Enron (non-DP)
python -m privacy_auditing.train \
  --model_version=gemma3_1b \
  --work_dir=/path/to/output \
  --max_steps=10000

# DP fine-tuning
python -m privacy_auditing.train \
  --model_version=gemma3_1b \
  --work_dir=/path/to/output \
  --enable_dp=True \
  --epsilon=8.0 \
  --delta=1e-5 \
  --max_steps=10000

# LoRA fine-tuning
python -m privacy_auditing.train \
  --model_version=gemma3_1b \
  --work_dir=/path/to/output \
  --lora_rank=64 \
  --max_steps=10000
```

### Privacy Auditing

> **Note:** Before running, implement `audit_lib.load_pretrained_model` to load
> the pretrained (base) Gemma 3 checkpoint for your environment. It currently
> raises `NotImplementedError`, and is shared by all scripts that load the
> pretrained model (`train.py`, `train_on_synthetic.py`, `model_audit.py`, and
> LoRA loading in `eval.py` / `generate_synthetic_data.py`).

```bash
python -m privacy_auditing.model_audit \
  --model_version=gemma3_1b \
  --canary_path=/path/to/canaries \
  --ckpt_path=/path/to/output/checkpoint \
  --work_dir=/path/to/audit_output
```

### Bigram Audit

```bash
python -m privacy_auditing.bigram_audit \
  --canary_path=/path/to/canaries \
  --canary_name=my_canaries \
  --synthetic_data_path=/path/to/synthetic_data.bagz \
  --output_dir=/path/to/bigram_output
```

### Synthetic Data

```bash
# Generate synthetic data from a fine-tuned model
python -m privacy_auditing.generate_synthetic_data \
  --model_version=gemma3_1b \
  --ckpt_path=/path/to/checkpoint \
  --work_dir=/path/to/synthetic_output

# Consolidate outputs from parallel generation
python -m privacy_auditing.consolidate_synthetic_data \
  --base_dir=/path/to/synthetic_output \
  --xids=XID1,XID2

# Train on synthetic data
python -m privacy_auditing.train_on_synthetic \
  --model_version=gemma3_1b \
  --synthetic_data_dir=/path/to/synthetic_output/XID1/consolidated_synthetic_data.bagz \
  --work_dir=/path/to/synth_train_output \
  --learning_rate=1e-4
```

### Evaluation

```bash
# Evaluate perplexity on held-out validation set
python -m privacy_auditing.eval \
  --model_version=gemma3_1b \
  --ckpt_path=/path/to/checkpoint \
  --work_dir=/path/to/eval_output
```

## Repository Structure

```
privacy_auditing/
├── audit_lib.py           # Core auditing utilities and model loading
├── bigram_audit.py        # Bigram baseline auditing script
├── bigram_audit_lib.py    # Bigram auditing library
├── consolidate_synthetic_data.py  # Merge synthetic data shards
├── dp_trainer.py          # DP-SGD fine-tuning trainer
├── eval.py                # Evaluation script
├── generate_synthetic_data.py  # Synthetic data generation
├── filesystem.py          # File I/O helpers (tf.io.gfile with stdlib fallback)
├── metrics.py             # Privacy metric computation
├── model_audit.py         # Model-based privacy auditing
├── poisson_dataset.py     # Poisson sampling dataset
├── prepare_data.py        # Dataset loading and preprocessing
├── train.py               # Main training script
└── train_on_synthetic.py  # Training on synthetic data
```

## Citation

If you use this code, please cite our paper:
```
[Citation to be added upon publication]
```

## License

Apache 2.0 — see [LICENSE](https://github.com/google-research/google-research/blob/master/LICENSE).
