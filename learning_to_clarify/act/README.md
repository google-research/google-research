# Learning to Clarify - Multi-turn Conversations with
Action-Based Contrastive Self-Training (ACT)

This is the official repository for the paper --
[Learning to Clarify - Multi-turn Conversations with
Action-Based Contrastive Self-Training](https://arxiv.org/pdf/2406.00222).

## Requirements

*   It is tested under Debian 5.10.209-2 (2024-01-31) x86_64 GNU/Linux and
    Python 3.10.14 environment.
*   To install requirements: `pip install -r requirements.txt`.
*   The Dockerfile can also be used with the nvidia-docker-runtime to generate
a compatible environment

## Datasets

To add a dataset, follow these instructions: * Add the dataset path in
`configs/base_config.py`. * Implement a dataset class in
`dataset_loaders/{new_dataset}.py` and import it in
`dataset_loaders/__init__.py`. * Add the new dataset into the
`get_dataset_loader` function in `utils/data_util.py`. * [Optional] Add few-shot
prompts for the new dataset in `datasets/few_shot_prompts.py`.

The datasets are available at `gs://jiefengc-tf-datasets/llm_dataset`.

## Models

The current codebase supports the Gemma 2 models.
The Gemma 2 decoder is implemented in `./models/gemma_decoder.py`.

## ACT Training

We implement ACT training in `run_act.py`.

To fine-tune gemma-2b using ACT on the PACIFIC
benchmark (using a tiny 1K training dataset), run the following command:

`ACCELERATE_LOG_LEVEL=info accelerate launch --config_file src/act/utils/deepspeed_zero3.yaml -m act.scripts.run_act --config=gs://madeka-learning-to-clarify/tiny_config_preftest.json`

## ACT Data

The methods in this module will primarily support the [Vertex AI Fine Tuning
Conversational Format](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning-about)
. This is done in order to ensure that data is compatible with the other APIS
available for tuning chat conversations on GCP.

## Primary Classes

The primary classes provided in this method are `ACTDataset` that takes a
dataset in the Vertex format and prepares it in a form that the HuggingFace
based `ACTTrainer` can use.

The syntax should be as follows:

```python
train_dataset = ACTDataset(
      train, config.training_config.target_label,
      config.training_config.icl_examples, preference_model,
      class_balance=config.training_config.class_balance,
      is_preference=config.training_config.is_preference,
  ).prepare_datasets()
```

Alternately, if one would like to load a dataset from a config - please use
the utility function `get_datasets_from_config` in `data.utils`.