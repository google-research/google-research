# Extended Transformer Construction (ETC)

This repository contains the core layers and fine-tuning code for "ETC: Encoding
Long and Structured Inputs in Transformers" (https://arxiv.org/abs/2004.08483),
published in EMNLP 2020.

To cite this work, please use:

```
@inproceedings{ainslie2020etc,
  title={ETC: Encoding Long and Structured Data in Transformers},
  author={Joshua Ainslie and Santiago Onta{\~n}{\'o}n and Chris Alberti and Vaclav Cvicek and Zachary Fisher and Philip Pham and Anirudh Ravula and Sumit Sanghai and Qifan Wang and Li Yang},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020)},
  year={2020}
}
```

## Requirements

The codebase requires Python 3. To download the code and install dependencies,
you can run a command like the following (ideally in a fresh Python environment,
e.g. from [virtualenv](https://pypi.org/project/virtualenv/)):

```
svn export https://github.com/google-research/google-research/trunk/etcmodel
pip install -r etcmodel/requirements.txt

# If using Google Cloud TPUs:
pip install cloud-tpu-client
```

Unit tests can be run via:

```
python -m unittest discover -s etcmodel -p '*_test.py'
```

When running the unit tests and all python commands mentioned later, the current
working directory must be the *parent* folder of the `etcmodel` folder.

## Code Structure

The code is structured as follows:

*   `layers` contains the core ETC Keras layers, compatible with both TF1 and
    TF2.
*   `models` contains modeling and example generation code for fine-tuning on
    the datasets in the paper. This code uses the `TPUEstimator` TF1 API like
    the original BERT codebase but depends on the `layers` code underneath.
*   `tensor_utils.py` and `feature_utils.py` contain general utilities for
    transforming tensors and creating ETC attention/mask features, respectively.

## Pre-trained Models

We release checkpoints for the following pre-trained models, all of which were
trained using CPC and hard g2l masking with one global token per natural
language sentence:

*   **[etc_base_2x_pretrain](https://storage.googleapis.com/gresearch/etcmodel/checkpoints/etc_base_2x_pretrain.zip)** -
    uses our default "base" configuration, except it was pre-trained for twice
    the number of epochs as other "base" configurations in the paper. This model
    was pre-trained from random initialization (like all the other "base" model
    in the paper) instead of lifting from RoBERTa weights like the "etc_large"
    models below.
*   **[etc_base_2x_pretrain_42relmaxdist](https://storage.googleapis.com/gresearch/etcmodel/checkpoints/etc_base_2x_pretrain_42relmaxdist.zip)** -
    uses the same configuration as `etc_base_2x_pretrain`, except with
    `relative_pos_max_distance` set to 42 instead of 12, providing some improved
    quality with additional compute cost. Although this checkpoint wasn't part
    of the ablations for the ETC paper itself, it was used in "Stepwise
    Extractive Summarization and Planning with Structured Transformers
    (https://arxiv.org/abs/2010.02744).
*   **[etc_large](https://storage.googleapis.com/gresearch/etcmodel/checkpoints/etc_large.zip)** -
    uses our "large" configuration and was initialized from RoBERTa weights.
    This was used for our best HotpotQA and WikiHop models.
*   **[etc_large_split_punctuation](https://storage.googleapis.com/gresearch/etcmodel/checkpoints/etc_large_split_punctuation.zip)** -
    uses the same configuration as `etc_large`, but expects text to be split
    into words via BERT's `BasicTokenizer` before being tokenized into
    SentencePieces. This has the effect of splitting punctuation into standalone
    tokens even if there isn't separating whitespace. For example, `"The end."`
    would be split into `["The", "end", "."]` instead of `["The, "end", "▁."]`.
    This model was used for our best NQ and OpenKP models since these two
    datasets pre-tokenize in this kind of manner.

Note that `etc_base_2x_pretrain` and `etc_base_2x_pretrain_42relmaxdist` use the
BERT uncased English WordPiece vocabulary, whereas `etc_large` and
`etc_large_split_punctuation` use the RoBERTa SentencePiece model vocabulary.
The corresponding vocabularies are included in the checkpoint directories.

During pre-training, each sentence global token used the same token id `1` from
the same embedding table as the WordPiece/SentencePiece vocabulary. Accordingly,
example generation for fine-tuning datasets should also use token id `1` for
sentence-level global tokens.

## TPU Configuration

We generally use a 32-core TPU v3 Pod slice from Google Cloud Platform (GCP) for
fine-tuning with batch size 64. Evaluation can be done on a single cloud TPU
rather than a Pod slice. We will use the following variable names, which should
be configured according to your GCP project configuration:

```
export TRAIN_TPU_NAME=my-train-gcp-tpu-name
export EVAL_TPU_NAME=my-eval-gcp-tpu-name
export TPU_ZONE=my-gcp-tpu-zone
export GCP_PROJECT_NAME=my-gcp-project-name
```

Note that TPUs cannot load data from the local disk of a VM, so the data needs
to be copied to a GCP bucket for TPU training/eval.

Due to some differences between TPU configuration on GCP vs. internally, we
noticed some TPU out of memory issues for some datasets when using batch size
64. In this case, reducing the batch size to 32 may be necessary despite some
potential loss in model quality.

## GPU Configuration

Currently only single-GPU training is supported, and it may be helpful to enable
gradient checkpointing (e.g. with `grad_checkpointing_period=1`) to fit as a
large a batch size in memory as possible. Even so, gradient accumulation may be
necessary to accommodate large enough batch sizes, but we haven't currently
implemented this.

We've seen that [XLA compilation](https://www.tensorflow.org/xla) can help
reduce memory footprint and compute time on GPUs. To activate it, you can use
this shell command:

```
export TF_XLA_FLAGS=--tf_xla_auto_jit=2
```

## Natural Questions Fine-tuning

The following shows how to perform fine-tuning using Google Cloud Platform (GCP)
TPUs.

Set the paths where all the code and data exists in your configuration (replace
these with the appropriate folders in your installation):

```
export NQ_ORIGINAL_FOLDER=/path/to/nq/v1.0
export NQ_PROCESSED_FOLDER=~/data/nq
export NQ_GCP_BUCKET_PATH=gs://path/to/gcp/bucket
export BERT_VOCAB=/path/to/vocab_bert_uncased_english.txt
export ROBERTA_VOCAB=/path/to/vocab_gpt_for_etc.model
export PRETRAINED_CHECKPOINT_FOLDER=/path/to/checkpointfolder
export MODEL_OUTPUT_FOLDER=/path/to/save/model/to
```

`NQ_ORIGINAL_FOLDER` is the folder where you downloaded the NQ dataset.
`NQ_PROCESSED_FOLDER` should be an empty folder you created to store the result
of pre-processing the NQ data. `NQ_GCP_BUCKET_PATH` specifies the path to the
GCP bucket that the data is to be copied to for TPU usage. Set `BERT_VOCAB`,
`ROBERTA_VOCAB` and `PRETRAINED_CHECKPOINT_FOLDER` to point to the paths (maybe
within a GCP bucket) where the vocab files and ETC pre-trained checkpoints are
located.

Set up the Python environment as described in the "Requirements" section.

The first step is to pre-process the NQ data to generate the training and eval
sets for the model. To pre-process the data using the BERT WordPiece tokenizer,
use the following commands:

```
mkdir ${NQ_PROCESSED_FOLDER}/proc
python3 -m etcmodel.models.nq.preproc_nq --input_dir=${NQ_ORIGINAL_FOLDER} \
--output_dir=${NQ_PROCESSED_FOLDER}/proc --vocab_file=${BERT_VOCAB}
gsutil -m cp -R ${NQ_PROCESSED_FOLDER}/proc ${NQ_GCP_BUCKET_PATH}
```

To pre-process the data using the RoBERTa SentencePiece tokenizer, use the
following commands:

```
mkdir ${NQ_PROCESSED_FOLDER}/proc-sp
python3 -m etcmodel.models.nq.preproc_nq --input_dir=${NQ_ORIGINAL_FOLDER} \
--output_dir=${NQ_PROCESSED_FOLDER}/proc-sp --tokenizer_type=ALBERT \
--spm_model_path=${ROBERTA_VOCAB}
gsutil -m cp -R ${NQ_PROCESSED_FOLDER}/proc-sp ${NQ_GCP_BUCKET_PATH}
```

After the data is pre-processed, the next step is to generate the gold data
cache needed by the Natural Questions evaluation script:

```
python3 -m etcmodel.models.nq.make_gold_cache \
--gold_path=${NQ_ORIGINAL_FOLDER}/dev/nq-dev-??.jsonl.gz
gsutil cp ${NQ_ORIGINAL_FOLDER}/dev/cache ${NQ_GCP_BUCKET_PATH}
```

Finally, we launch the fine-tuning jobs. The following two calls will start
first the training job, and then the eval job. The eval job will evaluate the
different checkpoints saved by the train job and output data for Tensorboard,
and should be running concurrently with training.

```
python3 -m etcmodel.models.nq.run_nq \
--etc_config_file=${PRETRAINED_CHECKPOINT_FOLDER}/etc_config.json \
--output_dir=${MODEL_OUTPUT_FOLDER} \
--gold_cache_path=${NQ_GCP_BUCKET_PATH}/cache \
--train_precomputed_file="${NQ_GCP_BUCKET_PATH}/proc/nq-train*" \
--predict_precomputed_file="${NQ_GCP_BUCKET_PATH}/proc/nq-dev*" \
--train_num_precomputed=${NUM_TRAIN_INSTANCES} \
--init_checkpoint=${PRETRAINED_CHECKPOINT_FOLDER}/model.ckpt \
--do_train=true \
--use_tpu=true \
--tpu_name=${TRAIN_TPU_NAME} \
--tpu_zone=${TPU_ZONE} \
--gcp_project=${GCP_PROJECT_NAME} \
--grad_checkpointing_period=1

python3 -m etcmodel.models.nq.run_nq \
--etc_config_file=${PRETRAINED_CHECKPOINT_FOLDER}/etc_config.json \
--output_dir=${MODEL_OUTPUT_FOLDER} \
--gold_cache_path=${NQ_GCP_BUCKET_PATH}/cache \
--train_precomputed_file="${NQ_GCP_BUCKET_PATH}/proc/nq-train*" \
--predict_precomputed_file="${NQ_GCP_BUCKET_PATH}/proc/nq-dev*" \
--train_num_precomputed=${NUM_TRAIN_INSTANCES} \
--init_checkpoint=${PRETRAINED_CHECKPOINT_FOLDER}/model.ckpt \
--do_predict=true \
--use_tpu=true \
--tpu_name=${EVAL_TPU_NAME} \
--tpu_zone=${TPU_ZONE} \
--gcp_project=${GCP_PROJECT_NAME} \
--grad_checkpointing_period=1
```

Replace `${NQ_GCP_BUCKET_PATH}/proc` by `${NQ_GCP_BUCKET_PATH}/proc-sp` above to
use the RoBERTa SentencePiece tokenization data instead of the BERT WordPiece
tokenization.

## HotpotQA Fine-tuning

First, download the [HotpotQA dataset](https://hotpotqa.github.io/)
(`hotpot_train_v1.1.json`, `hotpot_dev_distractor_v1.json`) to
`${HOTPOTQA_DATA_DIR}`, and specify the following paths. Note that there is also
a fullwiki task, which we don't consider in the paper.

```
export HOTPOTQA_DATA_DIR=/path/to/HotpotQA
export HOTPOTQA_EXAMPLE_DIR=${HOTPOTQA_DATA_DIR}/examples
export HOTPOTQA_EXAMPLE_GCP_BUCKET=gs://path/to/gcp/bucket
export HOTPOTQA_OUTPUT_FOLDER=/path/to/HotpotQA/output/folder
mkdir -p ${HOTPOTQA_EXAMPLE_DIR}
```

Now generate the examples. Each of the two input files can be processed
separately and processing can also be sped up by increasing
`--direct_num_workers`, but note that beam's DirectRunner uses large amount of
memory for each process. `direct_num_workers=0` is for automatically using all
available processes.

```
python3 -m etcmodel.models.hotpotqa.generate_tf_examples_beam \
--input_json_filename=${HOTPOTQA_DATA_DIR}/hotpot_train_v1.1.json \
--output_tf_examples_file_path_prefix=${HOTPOTQA_EXAMPLE_DIR}/tf_examples_train \
--num_shards=100 \
--output_tf_examples_stat_filename=${HOTPOTQA_EXAMPLE_DIR}/tf_examples_train_stats.txt \
--vocab_file=${BERT_VOCAB} \
--spm_model_file="" \
--global_seq_length=256 \
--long_seq_length=4096 \
--is_training="true" \
--answer_encoding_method="span" \
--direct_num_workers=1

python3 -m etcmodel.models.hotpotqa.generate_tf_examples_beam \
--input_json_filename=${HOTPOTQA_DATA_DIR}/hotpot_dev_distractor_v1.json \
--output_tf_examples_file_path_prefix=${HOTPOTQA_EXAMPLE_DIR}/tf_examples_dev_distractor \
--num_shards=10 \
--output_tf_examples_stat_filename=${HOTPOTQA_EXAMPLE_DIR}/tf_examples_dev_distractor_stats.txt \
--vocab_file=${BERT_VOCAB} \
--spm_model_file="" \
--global_seq_length=256 \
--long_seq_length=4096 \
--is_training="false" \
--answer_encoding_method="span" \
--direct_num_workers=1

gsutil -m cp -R ${HOTPOTQA_EXAMPLE_DIR} ${HOTPOTQA_EXAMPLE_GCP_BUCKET}
```

In order to use the SentencePiece tokenizer, please set
`--spm_model_path=${ROBERTA_VOCAB}` and `--vocab_file=""`.

Finally, TPU fine-tuning jobs can be launched with commands like the following
(one job for training and another for concurrent eval):

```
python3 -m etcmodel.models.hotpotqa.run_finetuning \
--etc_config_file=${PRETRAINED_CHECKPOINT_FOLDER}/etc_config.json \
--init_checkpoint=${PRETRAINED_CHECKPOINT_FOLDER}/model.ckpt \
--output_dir=${HOTPOTQA_OUTPUT_FOLDER} \
--train_tf_examples_filepattern="${HOTPOTQA_EXAMPLE_GCP_BUCKET}/tf_examples_train-*-of-00100" \
--num_train_tf_examples=90431 \
--predict_tf_examples_filepattern="${HOTPOTQA_EXAMPLE_GCP_BUCKET}/tf_examples_dev_distractor-*-of-00010" \
--predict_gold_json_file=${HOTPOTQA_DATA_DIR}/hotpot_dev_distractor_v1.json \
--spm_model_file="" \
--vocab_file=${BERT_VOCAB} \
--max_long_seq_length=4096 \
--max_global_seq_length=256 \
--run_mode="train" \
--learning_rate=5e-5 \
--num_train_epochs=9 \
--train_batch_size=64 \
--predict_batch_size=16 \
--flat_sequence=False \
--use_tpu=True \
--grad_checkpointing_period=0 \
--tpu_name=${TRAIN_TPU_NAME} \
--tpu_zone=${TPU_ZONE} \
--gcp_project=${GCP_PROJECT_NAME} \
--num_tpu_cores=32

python3 -m etcmodel.models.hotpotqa.run_finetuning \
--etc_config_file=${PRETRAINED_CHECKPOINT_FOLDER}/etc_config.json \
--init_checkpoint=${PRETRAINED_CHECKPOINT_FOLDER}/model.ckpt \
--output_dir=${HOTPOTQA_OUTPUT_FOLDER} \
--train_tf_examples_filepattern="${HOTPOTQA_EXAMPLE_GCP_BUCKET}/tf_examples_train-*-of-00100" \
--num_train_tf_examples=90431 \
--predict_tf_examples_filepattern="${HOTPOTQA_EXAMPLE_GCP_BUCKET}/tf_examples_dev_distractor-*-of-00010" \
--predict_gold_json_file=${HOTPOTQA_DATA_DIR}/hotpot_dev_distractor_v1.json \
--spm_model_file="" \
--vocab_file=${BERT_VOCAB} \
--max_long_seq_length=4096 \
--max_global_seq_length=256 \
--run_mode="predict" \
--learning_rate=5e-5 \
--num_train_epochs=9 \
--train_batch_size=64 \
--predict_batch_size=16 \
--flat_sequence=False \
--use_tpu=True \
--grad_checkpointing_period=0 \
--tpu_name=${TRAIN_TPU_NAME} \
--tpu_zone=${TPU_ZONE} \
--gcp_project=${GCP_PROJECT_NAME} \
--num_tpu_cores=8
```

Where a 4x4 TPU with 32 cores is used for training and a 2x2 TPU with 8 cores is
used for eval. For other TPU topologies, the `num_tpu_cores` need to be adjusted
accordingly. The pre-trained model is using a WordPiece tokenizer, and for
SentencePiece tokenizer, `--spm_model_path=${ROBERTA_VOCAB}` and
`--vocab_file=""` need to be used. When using a larger model or larger batch
size, a non-zero `grad_checkpointing_period` may be needed to reduce the TPU
memory usage. `grad_checkpointing_period=0` means no gradient checkpointing is
used. Note that the above example is for training an ETC-base model as described
in the paper. For the ETC-large model, described in the paper, the flags that
need to be changed are as follows: `--spm_model_path=${ROBERTA_VOCAB}
--vocab_file="" --learning_rate=3e-5 --num_train_epochs=5 --train_batch_size=32
--grad_checkpointing_period=1`. And we also need to change
`${PRETRAINED_CHECKPOINT_FOLDER}` to ETC-large and examples to SentencePiece
version.

## WikiHop Fine-tuning

First, download the [WikiHop dataset](http://qangaroo.cs.ucl.ac.uk/)
(`train.json`, `dev.json`) to `${WIKIHOP_DATA_DIR}`, and specify the following
paths. Note that there is also a fullwiki task, which we don't consider in the
paper.

```
export WIKIHOP_DATA_DIR=/path/to/WikiHop/
export WIKIHOP_EXAMPLE_PATH=gs://path/to/gcp/bucket
export WIKIHOP_MODEL_OUTPUT=/path/to/WikiHop/model/output
```

Download NLTK data.

```
python3
>> import nltk
>> nltk.download('punkt')
```

Now generate the examples.

```
python3 -m etcmodel.models.wikihop.generate_tf_examples \
--vocab_file=${BERT_VOCAB} \
--spm_model_path="" \
--input_train_json_filepath=${WIKIHOP_DATA_DIR}/train.json \
--input_dev_json_filepath=${WIKIHOP_DATA_DIR}/dev.json \
--long_seq_len=4096 \
--global_seq_len=430 \
--max_num_sentences=200 \
--tokenizer_type=BERT \
--output_dir_path=${WIKIHOP_EXAMPLE_PATH}
```

In order to use the SentencePiece tokenizer, please set
`--tokenizer_type=ALBERT, --spm_model_path=${ROBERTA_VOCAB}` `and
--vocab_file=""`.

Finally, TPU fine-tuning jobs can be launched with commands like the following
(one job for training and another for concurrent eval):

```
python3 -m etcmodel.models.wikihop.run_wikihop \
--source_model_config_file=${PRETRAINED_CHECKPOINT_FOLDER}/etc_config.json \
--output_dir=${WIKIHOP_MODEL_OUTPUT} \
--input_tf_records_path="${WIKIHOP_EXAMPLE_PATH}/train/tf_examples*" \
--init_checkpoint=${PRETRAINED_CHECKPOINT_FOLDER}/model.ckpt \
--do_train=True \
--candidate_ignore_hard_g2l=True \
--query_ignore_hard_g2l=True \
--enable_l2g_linking=True \
--long_seq_len=4096 \
--global_seq_len=430 \
--optimizer=adamw \
--learning_rate=4e-5 \
--train_batch_size=32 \
--num_train_examples=44000 \
--use_tpu=True \
--grad_checkpointing_period=0 \
--tpu_name=${TRAIN_TPU_NAME} \
--tpu_zone=${TPU_ZONE} \
--gcp_project=${GCP_PROJECT_NAME} \
--num_tpu_cores=32


python3 -m etcmodel.models.wikihop.run_wikihop \
--source_model_config_file=${PRETRAINED_CHECKPOINT_FOLDER}/etc_config.json \
--output_dir=${WIKIHOP_MODEL_OUTPUT} \
--input_tf_records_path="${WIKIHOP_EXAMPLE_PATH}/train/tf_examples*" \
--init_checkpoint=${PRETRAINED_CHECKPOINT_FOLDER}/model.ckpt \
--do_eval=True \
--candidate_ignore_hard_g2l=True \
--query_ignore_hard_g2l=True \
--enable_l2g_linking=True \
--long_seq_len=4096 \
--global_seq_len=430 \
--optimizer=adamw \
--learning_rate=4e-5 \
--eval_batch_size=8 \
--train_batch_size=32 \
--num_train_examples=44000 \
--use_tpu=True \
--grad_checkpointing_period=0 \
--tpu_name=${EVAL_TPU_NAME} \
--tpu_zone=${TPU_ZONE} \
--gcp_project=${GCP_PROJECT_NAME} \
--num_tpu_cores=8

```

Where a v3-32 TPU with 32 cores is used for training and a v3-8 TPU with 8 cores
is used for eval. For other TPU topologies, the `num_tpu_cores` need to be
adjusted accordingly. The pre-trained model is using a WordPiece tokenizer, and
for SentencePiece tokenizer, `--spm_model_path=${ROBERTA_VOCAB}` and
`--vocab_file=""` need to be used. When using a larger model or larger batch
size, a non-zero `grad_checkpointing_period` may be needed to reduce the TPU
memory usage. `grad_checkpointing_period=0` means no gradient checkpointing is
used. Note that the above example is for training an ETC-base model as described
in the paper. For the ETC-large model, described in the paper, the flags that
need to be changed are as follows: `--grad_checkpointing_period=1`. And we also
need to change `${PRETRAINED_CHECKPOINT_FOLDER}` to ETC-large and examples to
the SentencePiece version.

## OpenKP Fine-tuning

First, download the [OpenKP dataset](https://microsoft.github.io/msmarco/#kp)
(`OpenKPDev.jsonl`, `OpenKPTrain.jsonl`, `OpenKPEvalPublic.jsonl`) to
`${OPENKP_DATA_DIR}`, and specify the following paths.

```
export OPENKP_DATA_DIR=/path/to/OpenKP
export OPENKP_DEDUPED_DIR=${OPENKP_DATA_DIR}/deduped
export OPENKP_EXAMPLE_DIR=${OPENKP_DATA_DIR}/examples
export OPENKP_EXAMPLE_GCP_BUCKET=gs://path/to/gcp/bucket
mkdir -p ${OPENKP_DEDUPED_DIR}
mkdir -p ${OPENKP_EXAMPLE_DIR}
```

The original dataset contains several urls multiple times, so let us first
remove such duplicates.

```
# Keeps 6610 examples out of 6616.
python3 -m etcmodel.models.openkp.input_validate_and_dedup \
--input_file=${OPENKP_DATA_DIR}/OpenKPDev.jsonl \
--output_file=${OPENKP_DEDUPED_DIR}/OpenKPDev.jsonl

# Keeps 6613 examples out of 6614.
python3 -m etcmodel.models.openkp.input_validate_and_dedup \
--input_file=${OPENKP_DATA_DIR}/OpenKPEvalPublic.jsonl \
--output_file=${OPENKP_DEDUPED_DIR}/OpenKPEvalPublic.jsonl \
--is_eval=true

# Keeps 133724 examples out of 134894.
python3 -m etcmodel.models.openkp.input_validate_and_dedup \
--input_file=${OPENKP_DATA_DIR}/OpenKPTrain.jsonl \
--output_file=${OPENKP_DEDUPED_DIR}/OpenKPTrain.jsonl
```

Now generate the examples. Each of the three input files can be processed
separately and processing can also be sped up by increasing
`--direct_num_workers`, but note that beam's DirectRunner uses large amount of
memory for each process.

```
python3 -m etcmodel.models.openkp.generate_examples \
--input_patterns=${OPENKP_DEDUPED_DIR}/OpenKPDev.jsonl,${OPENKP_DEDUPED_DIR}/OpenKPEvalPublic.jsonl,${OPENKP_DEDUPED_DIR}/OpenKPTrain.jsonl \
--output_dir=${OPENKP_EXAMPLE_DIR} \
--output_num_shards=100 \
--long_max_length=4096 \
--global_max_length=512 \
--bert_vocab_path=${BERT_VOCAB} \
--do_lower_case=true

gsutil -m cp -R ${OPENKP_EXAMPLE_DIR} ${OPENKP_EXAMPLE_GCP_BUCKET}
```

In order to use the SentencePiece tokenizer, please set
`--spm_model_path=${ROBERTA_VOCAB}` and `--do_lower_case=false` in the last
command instead of `--bert_vocab_path`. A few examples don't parse successfully
(350 in `OpenKPTrain.jsonl`, 15 in `OpenKPDev.jsonl`), and are ignored.

Finally, TPU fine-tuning jobs can be launched with commands like the following
(one job for training and another for concurrent eval):

```
python3 -m etcmodel.models.openkp.run_finetuning \
--etc_config_file=${PRETRAINED_CHECKPOINT_FOLDER}/etc_config.json \
--output_dir=${MODEL_OUTPUT_FOLDER} \
--input_tfrecord="${OPENKP_EXAMPLE_GCP_BUCKET}/OpenKPTrain.tfrecord*" \
--init_checkpoint=${PRETRAINED_CHECKPOINT_FOLDER}/model.ckpt \
--do_train=true \
--batch_size=64 \
--num_train_examples=133374 \
--use_tpu=true \
--tpu_name=${TRAIN_TPU_NAME} \
--tpu_zone=${TPU_ZONE} \
--gcp_project=${GCP_PROJECT_NAME} \
--grad_checkpointing_period=1

python3 -m etcmodel.models.openkp.run_finetuning \
--etc_config_file=${PRETRAINED_CHECKPOINT_FOLDER}/etc_config.json \
--output_dir=${MODEL_OUTPUT_FOLDER} \
--input_tfrecord="${OPENKP_EXAMPLE_GCP_BUCKET}/OpenKPDev.tfrecord*" \
--init_checkpoint=${PRETRAINED_CHECKPOINT_FOLDER}/model.ckpt \
--eval_text_example_path=${OPENKP_EXAMPLE_GCP_BUCKET}/OpenKPDev_text_examples.jsonl \
--eval_fraction_of_removed_examples=0.002269288956 \
--do_eval=true \
--batch_size=64 \
--use_tpu=true \
--tpu_name=${EVAL_TPU_NAME} \
--tpu_zone=${TPU_ZONE} \
--gcp_project=${GCP_PROJECT_NAME} \
--grad_checkpointing_period=1
```

Note that we used a batch size of 64 by default internally, but due to out of
memory issues in external configuration, it may be necessary to reduce batch
size to 32 for "large" models on GCP (despite some resulting decrease in
metrics). Also, the best "large" run in the paper used learning rate 3e-5
instead of the default 5e-5 and a "max loss" setting which is not yet configured
in the public codebase.
