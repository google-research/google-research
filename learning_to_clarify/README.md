# Learning to Clarify (ACT)

Reference code for the paper [Learning to Clarify: Multi-turn Conversations with
Action-Based Contrastive Self-Training](https://arxiv.org/abs/2406.00222).

This is not an officially supported Google product.

If you use this code, please cite our paper:

```
@inproceedings{
chen2025learning,
title={Learning to Clarify: Multi-turn Conversations with Action-Based Contrastive Self-Training},
author={Maximillian Chen and Ruoxi Sun and Tomas Pfister and Sercan O Arik},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=SIE6VFps9x}
}
```

## Setup environment

Assume this file (README.md), Dockerfile, and the act subdirectory are all under
the `/home/myuser/staging/learning_to_clarify` directory.

### Download PACIFIC dataset (for example run)

```bash
cd /home/myuser/staging
git clone https://github.com/dengyang17/PACIFIC
```

### Convert PACIFIC dataset to Gemini 1.0 Pro format

```bash
cd /home/myuser/staging/learning_to_clarify
PYTHONPATH=/home/myuser/staging/learning_to_clarify/act/src python3 ./act/src/act/scripts/convert_pacific.py \
  --path=/home/myuser/staging/PACIFIC/data/pacific/train.json \
  --results_path=/home/myuser/staging/train.jsonl
PYTHONPATH=/home/myuser/staging/learning_to_clarify/act/src python3 ./act/src/act/scripts/convert_pacific.py \
  --path=/home/myuser/staging/PACIFIC/data/pacific/validation.json \
  --results_path=/home/myuser/staging/validation.jsonl
```

### Sample 20 entries from train.jsonl and validation.jsonl each for example run

For actual training you should use >= 50 samples for training, and all
validation samples.

```bash
cd /home/myuser/staging
shuf -n 20 train.jsonl > train_20samples.jsonl
shuf -n 20 validation.jsonl > validation_20samples.jsonl
```

## Build ACT

```bash
cd /home/myuser/staging/learning_to_clarify
docker build -t l2c .
```

## Generate preference data

A list of Google AI Studio Gemini models can be found at
https://ai.google.dev/gemini-api/docs/models.

Replace "YourApiKey" with your Google AI Studio key, which can be created at
https://aistudio.google.com/app/apikey.

```bash
mkdir -p /home/myuser/staging/output_dir/preference_data

docker run \
  -v /home/myuser/staging:/home/myuser/staging \
  -it \
  -e GOOGLE_API_KEY=YourApiKey l2c:latest generate-preference \
  --output_dir=/home/myuser/staging/output_dir/preference_data \
  --train_path=/home/myuser/staging/train_20samples.jsonl \
  --validation_path=/home/myuser/staging/validation_20samples.jsonl \
  --preference_model_id=gemini-2.0-flash-001 \
  --icl_examples=10
```

## Run SFT on Zephyr 7B beta model from Hugging Face

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/sft

docker run \
  -v /home/myuser/staging:/home/myuser/staging \
  -it \
  --runtime=nvidia --shm-size=1g -e NVIDIA_VISIBLE_DEVICES="all" \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e GOOGLE_API_KEY=YourApiKey l2c:latest run-sft \
  --output_dir=/home/myuser/staging/output_dir/model_output/sft \
  --train_path=/home/myuser/staging/output_dir/preference_data/train_preference.jsonl \
  --validation_path=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_token=YourHFTokenId \
  --policy_model_id=HuggingFaceH4/zephyr-7b-beta \
  --preference_model_id=gemini-2.0-flash-001 \
  --is_preference_task=True \
  --num_train_epochs=2
```

## Evaluate SFT

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/sft_eval

docker run \
  -v /home/myuser/staging:/home/myuser/staging \
  -it \
  --runtime=nvidia --shm-size=1g -e NVIDIA_VISIBLE_DEVICES="all" -e NVIDIA_DISABLE_REQUIRE=1 \
  -e GOOGLE_API_KEY=YourApiKey l2c:latest evaluate \
  --eval_data=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_path=/home/myuser/staging/output_dir/model_output/sft \
  --eval_result_output_path=/home/myuser/staging/output_dir/model_output/sft_eval/sft_eval.txt \
  --eval_sample_output_path=/home/myuser/staging/output_dir/model_output/sft_eval/sft_eval_samples.txt
```

## Run ACT on Zephyr 7B beta SFT model

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/act

docker run \
  -v /home/myuser/staging:/home/myuser/staging \
  -it \
  --runtime=nvidia --shm-size=1g -e NVIDIA_VISIBLE_DEVICES="all" -e NVIDIA_DISABLE_REQUIRE=1 \
  -e GOOGLE_API_KEY=YourApiKey l2c:latest run-act \
  --output_dir=/home/myuser/staging/output_dir/model_output/act \
  --train_path=/home/myuser/staging/output_dir/preference_data/train_preference.jsonl \
  --validation_path=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_path=/home/myuser/staging/output_dir/model_output/sft \
  --action_model_id=gemini-2.0-flash-001 \
  --simulator_model_id=gemini-2.0-flash-001 \
  --intent_model_id=gemini-2.0-flash-001 \
  --preference_model_id=gemini-2.0-flash-001 \
  --is_preference_task=True \
  --num_train_epochs=2
```

## Evaluate ACT

```bash
mkdir -p /home/myuser/staging/output_dir/model_output/act_eval

docker run \
  -v /home/myuser/staging:/home/myuser/staging \
  -it \
  --runtime=nvidia --shm-size=1g -e NVIDIA_VISIBLE_DEVICES="all" -e NVIDIA_DISABLE_REQUIRE=1 \
  -e GOOGLE_API_KEY=YourApiKey l2c:latest evaluate \
  --eval_data=/home/myuser/staging/output_dir/preference_data/validation_preference.jsonl \
  --policy_model_path=/home/myuser/staging/output_dir/model_output/act \
  --eval_result_output_path=/home/myuser/staging/output_dir/model_output/act_eval/act_eval.txt \
  --eval_sample_output_path=/home/myuser/staging/output_dir/model_output/act_eval/act_eval_samples.txt
```

## Contact

### Maintainers

* Max Chen (millian@google.com)
* Chris Baron (cgb@google.com)

### Contributors

* Vipin Nair (vipinvnair@google.com)
