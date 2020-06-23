# MobileBERT

This directory contains code for
[MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984).
MobileBERT is a thin version of BERT_LARGE, while equipped with bottleneck
structures and a carefully designed balance between self-attentions and
feed-forward networks.

![TensorFlow Requirement: 1.15](https://img.shields.io/badge/TensorFlow%20Requirement-1.15-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

## Pre-trained checkpoints

Download compressed files for pre-trained weights and `SavedModel`.

* MobileBert Optimized Uncased English:
[uncased_L-24_H-128_B-512_A-4_F-4_OPT](https://storage.googleapis.com/cloud-tpu-checkpoints/mobilebert/uncased_L-24_H-128_B-512_A-4_F-4_OPT.tar.gz)

* We provide both float32 and quantized int8 `SavedModel` for the Squad V1.1 that
are useful tf.lite conversion. Please checkout [mobilebert_squad_savedmodels](https://storage.googleapis.com/cloud-tpu-checkpoints/mobilebert/mobilebert_squad_savedmodels.tar.gz) for sequence length
at 384.

## Finetune with MobileBERT

```shell
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:/path/to/mobilebert
```

### SQuAD

Finetuning on SQuAD V1.1 and V2.0, we use the same script. The example script
finetunes the MobileBERT on SQuAD V1.1 dataset with TPU-v3-8.

```shell
export DATA_DIR=/tmp/mobilebert/data_cache/
export INIT_CHECKPOINT=/path/to/checkpoint/
export OUTPUT_DIR=/tmp/mobilebert/experiment/
python3 run_squad.py \
  --bert_config_file=config/uncased_L-24_H-128_B-512_A-4_F-4_OPT.json \
  --data_dir=${DATA_DIR} \
  --do_lower_case \
  --do_predict \
  --do_train \
  --doc_stride=128 \
  --init_checkpoint=${INIT_CHECKPOINT}/mobilebert.ckpt \
  --learning_rate=4e-05 \
  --max_answer_length=30 \
  --max_query_length=64 \
  --max_seq_length=384 \
  --n_best_size=20 \
  --num_train_epochs=5 \
  --output_dir=${OUTPUT_DIR} \
  --predict_file=/path/to/squad/dev-v1.1.json \
  --train_batch_size=32 \
  --train_file=/path/to/squad/train-v1.1.json \
  --use_tpu \
  --tpu_name=${TPU_NAME} \
  --vocab_file=${INIT_CHECKPOINT}/vocab.txt \
  --warmup_proportion=0.1
```

## Export MobileBERT to TF-Lite format.

```shell
export EXPORT_DIR='path/to/tflite'
python3 run_squad.py \
  --use_post_quantization=true \
  --activation_quantization=false \
  --data_dir=${DATA_DIR}  \
  --output_dir=${OUTPUT_DIR} \
  --vocab_file=${INIT_CHECKPOINT}/vocab.txt \
  --bert_config_file=config/uncased_L-24_H-128_B-512_A-4_F-4_OPT.json \
  --train_file=/path/to/squad/train-v1.1.json \
  --export_dir=${EXPORT_DIR}
```

## MobileBERT Pre-training & Distillation

### Data processing

We use the exact identical data processing script to prepare pre-training data
as BERT. Please use the BERT data processing script
[create_pretraining_data.py](https://github.com/google-research/bert/blob/master/create_pretraining_data.py)
to obtain tfrecords. Regarding the datasets, please see
https://github.com/google-research/bert#pre-training-data.

### Distillation

We conducted distillation process on the pre-training data with TPU-v3-256.

```shell
export TEACHER_CHECKPOINT=/path/to/checkpoint/
export OUTPUT_DIR=/tmp/mobilebert/experiment/
python3 run_pretraining.py \
  --attention_distill_factor=1 \
  --bert_config_file=config/uncased_L-24_H-128_B-512_A-4_F-4_OPT.json \
  --bert_teacher_config_file=config/uncased_L-24_H-1024_B-512_A-4.json \
  --beta_distill_factor=5000 \
  --distill_ground_truth_ratio=0.5 \
  --distill_temperature=1 \
  --do_train \
  --first_input_file=/path/to/pretraining_data \
  --first_max_seq_length=128 \
  --first_num_train_steps=0 \
  --first_train_batch_size=4096 \
  --gamma_distill_factor=5 \
  --hidden_distill_factor=100 \
  --init_checkpoint=${TEACHER_CHECKPOINT} \
  --input_file=path/to/pretraining_data \
  --layer_wise_warmup \
  --learning_rate=0.0015 \
  --max_predictions_per_seq=20 \
  --max_seq_length=512 \
  --num_distill_steps=240000 \
  --num_train_steps=500000 \
  --num_warmup_steps=10000 \
  --optimizer=lamb \
  --output_dir=${OUTPUT_DIR} \
  --save_checkpoints_steps=10000 \
  --train_batch_size=2048 \
  --use_einsum \
  --use_summary \
  --use_tpu \
  --tpu_name=${TPU_NAME} \
```

### Run Quantization-aware-training with Squad

After we have distilled the pre-trained mobile bert, we can insert fake quant
nodes for quantization-aware-training:

```shell
export DATA_DIR=/tmp/mobilebert/data_cache/
export INIT_CHECKPOINT=/path/to/checkpoint/
export OUTPUT_DIR=/tmp/mobilebert/experiment/
python3 run_squad.py \
  --bert_config_file=config/uncased_L-24_H-128_B-512_A-4_F-4_OPT_QAT.json \
  --data_dir=${DATA_DIR} \
  --do_lower_case \
  --do_predict \
  --do_train \
  --doc_stride=128 \
  --init_checkpoint=${INIT_CHECKPOINT}/mobilebert.ckpt \
  --learning_rate=4e-05 \
  --max_answer_length=30 \
  --max_query_length=64 \
  --max_seq_length=384 \
  --n_best_size=20 \
  --num_train_epochs=5 \
  --output_dir=${OUTPUT_DIR} \
  --predict_file=/path/to/squad/dev-v1.1.json \
  --train_batch_size=32 \
  --train_file=/path/to/squad/train-v1.1.json \
  --use_tpu \
  --tpu_name=${TPU_NAME} \
  --vocab_file=${INIT_CHECKPOINT}/vocab.txt \
  --warmup_proportion=0.1
  --use_quantized_training=true
```

## Export an integer-only MobileBERT to TF-Lite format.

```shell
export EXPORT_DIR='path/to/tflite'
python3 run_squad.py \
  --use_quantized_training=true \
  --use_post_quantization=true \
  --activation_quantization=true \
  --data_dir=${DATA_DIR}  \
  --output_dir=${OUTPUT_DIR} \
  --vocab_file=${INIT_CHECKPOINT}/vocab.txt \
  --bert_config_file=config/uncased_L-24_H-128_B-512_A-4_F-4_OPT_QAT.json \
  --train_file=/path/to/squad/train-v1.1.json \
  --export_dir=${EXPORT_DIR}
```
