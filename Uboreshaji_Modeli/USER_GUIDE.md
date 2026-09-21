# Uboreshaji Modeli: Multimodal SFT User Guide

This guide provides exhaustive instructions for configuring, launching, and
optimizing fine-tuning experiments across **Vision**, **Text**, and **Speech**
modalities using the modular composed strategy pattern framework.

--------------------------------------------------------------------------------

## 1. Core Orchestration Workflow

The entry point `main.py` acts as a thin launcher that overrides configurations
via CLI flags. The actual orchestration is handled by the universal
`main_lib.py` which:

1.  Resolves the target device (CPU, CUDA, or TPU).
2.  Instantiates the composed `ModelEngine` via factory assembler.
3.  Loads weights, processors, and wraps models with PEFT/LoRA adapters.
4.  Loads datasets (and applies local caching/streaming locks).
5.  Resolves the pluggable task `TrainerStrategy` and delegates execution.

## 2. General Fine-Tuning Recipe

To fine-tune a model using the `Uboreshaji_Modeli` framework, follow these
general steps:

1.  **Select/Prepare your Dataset**: Ensure your dataset is formatted correctly for the target modality (e.g., COCO JSON for detection, Hugging Face `datasets` with chat history for Text SFT).
2.  **Configure Hyperparameters**: Open the relevant config file in the `configs/` directory (e.g., `gemma4_config.py`). Adjust key knobs:
    *   `cfg.training.learning_rate`: Start small (e.g., `2e-5`).
    *   `cfg.training.batch_size`: Per-device batch size.
    *   `cfg.training.gradient_accumulation_steps`: Increase to simulate larger batches.
    *   `cfg.training.use_lora`: Enable for Parameter-Efficient Tuning.
3.  **Launch**: Execute the training script using the modality-specific commands below.

--------------------------------------------------------------------------------

## 3. Supported Modalities & Launch Command Guide

### A. Object Detection (OWL-v2)

Runs the set criterion Hungarian loss trainer loop.

*   **Local run:**

    ```bash
    python main.py \
      --config=configs/example_config.py \
      --model_id="google/owlv2-base-patch16-ensemble" \
      --dataset_path="/path/to/my/hf_dataset" \
      --output_dir="/tmp/outputs"
    ```


### B. Generative Vision SFT (Gemma 4 VLM)

Trains VLM vision-language models to locate items using bounding box coordinates
or outline segmentation polygons.

*   **Local / Single-GPU run:**

    ```bash
    python main.py \
      --config=configs/gemma_config.py \
      --output_dir="/tmp/outputs"
    ```


### C. Generative Text SFT (Gemma 3/4 Causal SFT)

Fine-tunes chat or instruction models, automatically masking out prompt/user
tokens to compute SFT training loss strictly on the assistant output tokens.

*   **Local / Distributed run:**

    ```bash
    python main.py \
      --config=configs/gemma_text_sft_config.py \
      --output_dir="/tmp/outputs"
    ```


### D. Speech ASR and Audio SFT (Whisper, MMS, Gemma Audio)

Supports sequence-to-sequence transcript generation (Whisper) and Wav2Vec2 CTC
SFT using dynamic padding.

*   **Local run:**

    ```bash
    python main.py \
      --config=configs/mms_asr_config.py \
      --output_dir="/tmp/outputs"
    ```


--------------------------------------------------------------------------------

## 4. Optimization Config Guidelines

The composed strategies leverage advanced performance and memory optimizations:

### 💎 Precision Options

Precision is configured via `cfg.training.precision`. Supported values:

-   `"bf16"`: Enforces Brain Float 16 mixed precision. Requires CUDA.
-   `"fp16"`: Enforces Float 16 mixed precision.
-   `"fp32"`: Enforces standard Float 32 precision. Safely falls back to
    `"fp32"` if CUDA is unavailable.

### 💎 Dataset Caching vs Streaming

-   **Caching (Standard)**: Suitable for small or medium datasets. On multi-node
    DDP runs, Rank 0 blocks and downloads the dataset, caching it locally to
    disk. All other ranks yield via `dist.barrier()` and read from local cache,
    eliminating network file I/O locks.
-   **Streaming (`cfg.dataset.streaming = True`)**: Mandatory for infinite
    speech or very large datasets. It streams raw bytes on-the-fly (using
    `IterableDataset`) without local file caching. It automatically bypasses
    distributed cache locks, allowing all ranks to establish independent stream
    connections instantly.


### 💎 Memory Fragmentation Bypass (`expandable_segments`)

During large-backbone Gemma SFT runs, GPU memory fragmentations can cause fatal
OutOfMemory (OOM) errors.

*   The framework automatically injects `PYTORCH_CUDA_ALLOC_CONF =
    "expandable_segments:True"` eagerly on the very first line of execution
    before any PyTorch context is initialized, enabling PyTorch to allocate
    segments dynamically and preventing up to 40% of fragmentation OOMs.

### 💎 Detection Formats & Token Economy

When fine-tuning Gemma VLMs for detection, the format of the output string
matters significantly for memory efficiency, especially in images with many
objects.

*   **JSON Format (`cfg.detection_format = 'json'`)**: Native to Gemini/Gemma.
    Verbose. Each bracket, quote, and coordinate consumes tokens (~15-20 tokens
    per object). Best for standard scenes.
*   **Loc Format (`cfg.detection_format = 'loc'`)**: Highly compact. The
    framework injects `<locYYYY>` as **special custom tokens** (1 token per
    coordinate). This reduces footprint to ~6 tokens per object (4 locations +
    label + separator).

> [!TIP]
> If your dataset has "crowded" scenes with hundreds of objects, use `loc`
> format to drastically reduce sequence length and prevent Activation OOMs.

### 💎 Parameter-Efficient Adapters (PEFT/LoRA)

Configured via `cfg.training.use_lora = True`:

-   Target modules are parsed dynamically. Target sub-modules (like custom Gemma
    wrappers `Gemma4ClippableLinear`) are automatically mapped to underlying
    supported `nn.Linear` references.
-   The DDP strategy automatically handles checkpoints so that **only** adapter
    weights are saved, saving up to 90% of storage bandwidth on massive Gemma
    checkpoints.


--------------------------------------------------------------------------------

## 5. TPU Execution (PyTorch/XLA Integration)

The framework supports dynamic execution routing to TPUs via PyTorch/XLA.

### A. Configuration File Setup

To configure an experiment for TPU execution, edit your configuration file:

```python
# Force routing to TPU device
cfg.training.device = "tpu"
```

### B. Launching TPU Jobs

For open-source TPU environments (e.g. Cloud TPU VMs):

```bash
python main.py \
  --config=configs/gemma_text_sft_config.py \
  --output_dir="/tmp/outputs"
```


--------------------------------------------------------------------------------

## 6. Known Limitations & TPU-Specific Caveats

### ⚠️ Dynamic Audio Dataset Loading (Freezes on TPUs, Works on GPUs)

When attempting to fine-tune speech/audio models (like Whisper, MMS, or Gemma
Audio) on **TPU slices**, training runs may freeze during dynamic audio
decoding.

