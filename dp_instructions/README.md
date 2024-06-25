# Privacy-preserving Instructions for Aligning LLMs

This repo contains code for reproducing the results in [Privacy-preserving Instructions for Aligning Large Language Models](https://arxiv.org/abs/2402.13659).

## Overview

This repo implements:
1. DP fine-tuning of LLaMA 7B/13B models on Chatbot Arena instructions.
2. Sample from fine-tuned models to generate initial synthetic instructions.
3. Resample the initial synthetic instructions with DP histogram.
4. Query OpenAI's API to label real/synthetic instructions.
5. Supervised fine-tuning of LLaMA 7B/13B models with (instruction, response) pairs.

For running Reinforcement Learning with Human Feedback (RLHF) with PPO, we use the off-the-shelf implementation from [Huggingface TRL](https://github.com/huggingface/trl).

## Pointers to the Main Components

### Setup

Please first install the required packages.
```
pip install -r requirements.txt
```

### Pre-process LMSYS-1M dataset

This step helps you build the Chatbot Arena instructions used in our paper. Please first apply access for the [LMSYS-Chat-1M dataset](https://huggingface.co/datasets/lmsys/lmsys-chat-1m).

Then, simply run the following script. The dataset is a huggingface ```DatasetDict``` object and will be saved in ```dp_finetuning/data/```.
```
python data_preprocess.py
```

### Running Experiments

For 1, 2, and 5, please follow the README.md in ```dp_finetuning```.

For 3, please follow the README.md in ```resample_with_dp_histogram```.

For 4, please follow the README.md in ```query_gpt35_for_annotation```.



