# ReadTwice model

This repository contains the code for `ReadTwice: Reading Very Large Documents with Memories` paper.

Some of the functions were originally implemented for [ETC](https://github.com/google-research/google-research/tree/master/etcmodel) project.

## Requirements

```
svn export https://github.com/google-research/google-research/trunk/readtwice
pip install -r readtwice/requirements.txt
```

WARNING: Currently, the code relies on some ops (namely, `cross_replica_concat`), which require TPU (or CPU). However, it should be possible to adjust the code to make it GPU-friendly.

```
pip install cloud-tpu-client
```

Unit tests can be run via:

```bash
python -m unittest discover -s readtwice -p '*_test.py'
```
When running the unit tests and all python commands mentioned later, the current working directory must be the parent folder of the `readtwice` folder.

## Pre-trained Models

We release a **[pre-trained checkpoint](http://storage.googleapis.com/gresearch/readtwice/readtwice.tar.gz)** that can be used to reproduce experimental results.

We use the same vocabulary as RoBERTa model where the equivalent SentencePiece tokenizer was provided by [BERT Seq2Seq](https://github.com/google-research/google-research/tree/master/bertseq2seq) project.

## Fine-tuning

Below as detailed instructions on reproducing main results from the paper.

First, download the [SentencePiece vocabulary](https://storage.googleapis.com/berts2s-tokenizers-tacl20/vocab_gpt.model), and download and untar the [pre-trained checkpoint](http://storage.googleapis.com/gresearch/readtwice/readtwice.tar.gz). Set the following paths accordingly.

```bash
export PRETRAINED_MODEL_DIR=gs://path/to/directory/with/pretrained/model
export CONFIG_PATH=${PRETRAINED_MODEL_DIR}/read_it_twice_bert_config.json
export PRETRAINED_MODEL_CHECKPOINIT=${PRETRAINED_MODEL_DIR}/model.ckpt-1000000

export SPM_MODEL_PATH=/path/to/vocab_gpt.model
export NLTK_DATA_PATH=/tmp/nltk_dir
```

### HotpotQA

First, download the [HotpotQA dataset](https://hotpotqa.github.io/) (`hotpot_train_v1.1.json`, `hotpot_dev_distractor_v1.json`) to `${HOTPOTQA_DATA_DIR}`, and specify the following paths.

```bash
export HOTPOTQA_DATA_DIR=/path/to/HotpotQA
export HOTPOTQA_EXAMPLE_DIR=${HOTPOTQA_DATA_DIR}/examples
export HOTPOTQA_EXAMPLE_GCP_BUCKET=gs://path/to/gcp/bucket
export HOTPOTQA_OUTPUT_FOLDER=gs://path/to/HotpotQA/output/folder
mkdir -p ${HOTPOTQA_EXAMPLE_DIR}
```

Second, generate TFExamples files using the following commands and copy files to GCS.

```bash
python -m readtwice.models.hotpot_qa.preprocess \
--spm_model_path=${SPM_MODEL_PATH} \
--input_file=${HOTPOTQA_DATA_DIR}/hotpot_dev_distractor_v1.json \
--output_prefix=${HOTPOTQA_EXAMPLE_DIR}/valid \
--nltk_data_path=${NLTK_DATA_PATH}

python -m readtwice.models.hotpot_qa.preprocess \
--spm_model_path=${SPM_MODEL_PATH} \
--input_file=${HOTPOTQA_DATA_DIR}/hotpot_train_v1.1.json \
--output_prefix=${HOTPOTQA_EXAMPLE_DIR}/train \
--generate_answers \
--nltk_data_path=${NLTK_DATA_PATH}

gsutil -m cp ${HOTPOTQA_EXAMPLE_DIR}/* ${HOTPOTQA_EXAMPLE_GCP_BUCKET}
gsutil -m cp ${HOTPOTQA_DATA_DIR}/hotpot_dev_distractor_v1.json ${HOTPOTQA_EXAMPLE_GCP_BUCKET}
```

Finally, launch a fine-tuning using a pre-training.

```bash
python -m readtwice.models.hotpot_qa.run_finetuning \
--read_it_twice_bert_config_file=${CONFIG_PATH} \
--input_file=${HOTPOTQA_EXAMPLE_GCP_BUCKET}/train.tfrecord-* \
--output_dir=${HOTPOTQA_OUTPUT_FOLDER} \
--init_checkpoint=${PRETRAINED_MODEL_CHECKPOINIT} \
--enable_side_inputs \
--cross_block_attention_mode=doc \
--do_train \
--nodo_eval \
--optimizer=adamw \
--learning_rate=3e-05 \
--num_train_epochs=6 \
--warmup_proportion=0.1 \
--learning_rate_schedule=inverse_sqrt \
--poly_power=1 \
--start_warmup_step=0 \
--save_checkpoints_steps=5000 \
--iterations_per_loop=1000 \
--nouse_one_hot_embeddings \
--use_tpu \
--tpu_job_name=??? \
--num_tpu_cores=16 \
--num_tpu_tasks=1 \
--decode_top_k=40 \
--decode_max_size=10 \
--tpu_name=??? \
--cross_attention_top_k=100

python -m readtwice.models.hotpot_qa.run_finetuning \
--read_it_twice_bert_config_file=${CONFIG_PATH} \
--input_file=${HOTPOTQA_EXAMPLE_GCP_BUCKET}/valid.tfrecord-* \
--output_dir=${HOTPOTQA_OUTPUT_FOLDER} \
--init_checkpoint=${PRETRAINED_MODEL_CHECKPOINIT} \
--enable_side_inputs \
--cross_block_attention_mode=doc \
--nodo_train \
--do_eval \
--optimizer=adamw \
--learning_rate=3e-05 \
--num_train_epochs=6 \
--warmup_proportion=0.1 \
--learning_rate_schedule=inverse_sqrt \
--poly_power=1 \
--start_warmup_step=0 \
--save_checkpoints_steps=5000 \
--iterations_per_loop=1000 \
--nouse_one_hot_embeddings \
--use_tpu \
--tpu_job_name=??? \
--num_tpu_cores=16 \
--num_tpu_tasks=1 \
--decode_top_k=40 \
--decode_max_size=10 \
--tpu_name=??? \
--cross_attention_top_k=100
```

WARNING: Evaluating the output requires additional steps outlined in the Appendix of the paper.

### TriviaQA

First, download data from [TriviaQA official website](https://nlp.cs.washington.edu/triviaqa), and specify the following paths.

```bash
export TRIVIAQA_DATA_DIR=/path/to/TriviaQA
export TRIVIAQA_EXAMPLE_DIR=${TRIVIAQA_DATA_DIR}/examples
export TRIVIAQA_EXAMPLE_GCP_BUCKET=gs://path/to/gcp/bucket
export TRIVIAQA_OUTPUT_FOLDER=gs://path/to/TriviaQA/output/folder
mkdir -p ${TRIVIAQA_EXAMPLE_DIR}
```

Second, generate TFExamples files using the following commands and copy files to GCS.

```bash
python -m readtwice.models.trivia_qa.preprocess \
--spm_model_path=${SPM_MODEL_PATH} \
--input_file=${TRIVIAQA_DATA_DIR}/qa/wikipedia-dev.json \
--wikipedia_dir=${TRIVIAQA_DATA_DIR}/evidence/wikipedia \
--web_dir=${TRIVIAQA_DATA_DIR}/evidence/web \
--output_prefix=${TRIVIAQA_EXAMPLE_DIR}/valid \
--nltk_data_path=${NLTK_DATA_PATH}

python -m readtwice.models.trivia_qa.preprocess \
--spm_model_path=${SPM_MODEL_PATH} \
--input_file=${TRIVIAQA_DATA_DIR}/qa/wikipedia-train.json \
--wikipedia_dir=${TRIVIAQA_DATA_DIR}/evidence/wikipedia \
--web_dir=${TRIVIAQA_DATA_DIR}/evidence/web \
--output_prefix=${TRIVIAQA_EXAMPLE_DIR}/train \
--generate_answers \
--nltk_data_path=${NLTK_DATA_PATH}

gsutil -m cp ${TRIVIAQA_EXAMPLE_DIR}/* ${TRIVIAQA_EXAMPLE_GCP_BUCKET}
gsutil -m cp ${TRIVIAQA_DATA_DIR}/qa/wikipedia-dev.json ${TRIVIAQA_EXAMPLE_GCP_BUCKET}
```

Finally, launch a fine-tuning using a pre-training.

```bash
python -m readtwice.models.trivia_qa.run_finetuning \
--read_it_twice_bert_config_file=${CONFIG_PATH} \
--input_file=${TRIVIAQA_EXAMPLE_GCP_BUCKET}/train.tfrecord-* \
--eval_json_path=${TRIVIAQA_EXAMPLE_GCP_BUCKET}/wikipedia-dev.json \
--output_dir=${TRIVIAQA_OUTPUT_FOLDER} \
--init_checkpoint=${PRETRAINED_MODEL_CHECKPOINIT} \
--enable_side_inputs \
--cross_block_attention_mode=doc \
--do_train \
--nodo_eval \
--optimizer=adamw \
--learning_rate=1e-05 \
--num_train_epochs=6 \
--warmup_proportion=0.1 \
--learning_rate_schedule=poly_decay \
--poly_power=1 \
--start_warmup_step=0 \
--save_checkpoints_steps=3000 \
--iterations_per_loop=200 \
--nouse_one_hot_embeddings \
--use_tpu \
--tpu_job_name=??? \
--num_tpu_cores=16 \
--num_tpu_tasks=1 \
--decode_top_k=8 \
--decode_max_size=20 \
--eval_data_split=valid \
--spm_model_path=${SPM_MODEL_PATH} \
--tpu_name=??? \
--cross_attention_top_k=100

python -m readtwice.models.trivia_qa.run_finetuning \
--read_it_twice_bert_config_file=${CONFIG_PATH} \
--input_file=${TRIVIAQA_EXAMPLE_GCP_BUCKET}/valid.tfrecord-* \
--eval_json_path=${TRIVIAQA_EXAMPLE_GCP_BUCKET}/wikipedia-dev.json \
--output_dir=${TRIVIAQA_OUTPUT_FOLDER} \
--init_checkpoint=${PRETRAINED_MODEL_CHECKPOINIT} \
--enable_side_inputs \
--cross_block_attention_mode=doc \
--nodo_train \
--do_eval \
--optimizer=adamw \
--learning_rate=1e-05 \
--num_train_epochs=6 \
--warmup_proportion=0.1 \
--learning_rate_schedule=poly_decay \
--poly_power=1 \
--start_warmup_step=0 \
--save_checkpoints_steps=3000 \
--iterations_per_loop=200 \
--nouse_one_hot_embeddings \
--use_tpu \
--tpu_job_name=??? \
--num_tpu_cores=16 \
--num_tpu_tasks=1 \
--decode_top_k=8 \
--decode_max_size=20 \
--eval_data_split=valid \
--spm_model_path=${SPM_MODEL_PATH} \
--tpu_name=??? \
--cross_attention_top_k=100
```

### NarrativeQA

First, download data from [NarrativeQA official website](https://github.com/deepmind/narrativeqa), and specify the following paths.

```bash
export NARRATIVEQA_DATA_DIR=/path/to/NarrativeQA
export NARRATIVEQA_EXAMPLE_DIR=${NARRATIVEQA_DATA_DIR}/examples
export NARRATIVEQA_EXAMPLE_GCP_BUCKET=gs://path/to/gcp/bucket
export NARRATIVEQA_OUTPUT_FOLDER=gs://path/to/NarrativeQA/output/folder
mkdir -p ${NARRATIVEQA_EXAMPLE_DIR}
```

Second, generate TFExamples files using the following commands and copy files to GCS.

```bash
python -m readtwice.models.narrative_qa.preprocess \
--spm_model_path=${SPM_MODEL_PATH} \
--input_qaps=${NARRATIVEQA_DATA_DIR}/qaps.csv \
--input_documents=${NARRATIVEQA_DATA_DIR}/documents.csv \
--data_split=valid \
--stories_dir=${NARRATIVEQA_DATA_DIR}/tmp/ \
--output_prefix=${NARRATIVEQA_EXAMPLE_DIR}/valid \
--nltk_data_path=${NLTK_DATA_PATH}


python -m readtwice.models.narrative_qa.preprocess \
--spm_model_path=${SPM_MODEL_PATH} \
--input_qaps=${NARRATIVEQA_DATA_DIR}/qaps.csv \
--input_documents=${NARRATIVEQA_DATA_DIR}/documents.csv \
--data_split=train \
--stories_dir=${NARRATIVEQA_DATA_DIR}/tmp/ \
--output_prefix=${OUTPUT_NARRATIVE_QA}/train \
--generate_answers \
--nltk_data_path=${NLTK_DATA_PATH}

gsutil cp ${NARRATIVEQA_DATA_DIR}qaps.csv ${NARRATIVEQA_EXAMPLE_DIR}
```

Finally, launch a fine-tuning using a pre-training.

```bash
python -m readtwice.models.trivia_qa.run_finetuning \
--read_it_twice_bert_config_file=${CONFIG_PATH} \
--input_file=${NARRATIVEQA_EXAMPLE_DIR}/train.tfrecord-* \
--input_qaps=${NARRATIVEQA_EXAMPLE_DIR}/qaps.csv \
--eval_data_split=valid \
--output_dir=${NARRATIVEQA_OUTPUT_FOLDER} \
--init_checkpoint=${PRETRAINED_MODEL_CHECKPOINIT} \
--enable_side_inputs \
--cross_block_attention_mode=doc \
--do_train \
--nodo_eval \
--optimizer=adamw \
--learning_rate=5e-06 \
--nosummary_enable_default_side_input \
--num_train_epochs=6 \
--warmup_proportion=0.1 \
--learning_rate_schedule=inverse_sqrt \
--poly_power=1 \
--start_warmup_step=0 \
--save_checkpoints_steps=15000 \
--spm_model_path=${SPM_MODEL_PATH} \
--iterations_per_loop=1000 \
--nouse_one_hot_embeddings \
--use_tpu \
--tpu_job_name=??? \
--num_tpu_cores=16 \
--num_tpu_tasks=1 \
--decode_top_k=40 \
--decode_max_size=10 \
--tpu_name=??? \
--cross_attention_top_k=100

python -m readtwice.models.trivia_qa.run_finetuning \
--read_it_twice_bert_config_file=${CONFIG_PATH} \
--input_file=${NARRATIVEQA_EXAMPLE_DIR}/valid.tfrecord-* \
--input_qaps=${NARRATIVEQA_EXAMPLE_DIR}/qaps.csv \
--eval_data_split=valid \
--output_dir=${NARRATIVEQA_OUTPUT_FOLDER} \
--init_checkpoint=${PRETRAINED_MODEL_CHECKPOINIT} \
--enable_side_inputs \
--cross_block_attention_mode=doc \
--nodo_train \
--do_eval \
--optimizer=adamw \
--learning_rate=5e-06 \
--nosummary_enable_default_side_input \
--num_train_epochs=6 \
--warmup_proportion=0.1 \
--learning_rate_schedule=inverse_sqrt \
--poly_power=1 \
--start_warmup_step=0 \
--save_checkpoints_steps=15000 \
--spm_model_path=${SPM_MODEL_PATH} \
--iterations_per_loop=1000 \
--nouse_one_hot_embeddings \
--use_tpu \
--tpu_job_name=??? \
--num_tpu_cores=16 \
--num_tpu_tasks=1 \
--decode_top_k=40 \
--decode_max_size=10 \
--tpu_name=??? \
--cross_attention_top_k=100
```

## Pre-training
Pre-training code is yet to be released. There are two main issues that needs to be resolved. First, the pre-training is using a custom TF operations to perform words and entities masking on a fly. Second, the data preprocessing is currently relies on an proprietary infrastructure. Meanwhile, we release the pre-training demo which while not executable, demonstrates the core implementation details.
