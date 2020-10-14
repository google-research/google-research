
# SMITH (Siamese Multi-depth Transformer-based Hierarchical Encoder) Model Code

This repository maintains the code of [SMITH (Siamese Multi-depth Transformer-based
Hierarchical Encoder) model](https://research.google/pubs/pub49617/). The model can be
used for the [long-form document matching](https://research.google/pubs/pub47856/)
task.


## Introduction
Many natural language processing and information retrieval problems can be
formalized as the task of semantic matching. Existing work in this area has
been largely focused on matching between short texts (e.g., question answering),
or between a short and a long text (e.g., ad-hoc retrieval). Semantic matching
between long-form documents, which has many important applications like news
recommendation, related article recommendation and document clustering, is
relatively less explored and needs more research effort. In recent years,
self-attention based models like Transformers and BERT have achieved
state-of-the-art performance in the task of text matching. These models,
however, are still limited to short text like a few sentences or one paragraph
due to the quadratic computational complexity of self-attention with respect to
input text length. To address this issue, we proposed the Siamese Multi-depth
Transformer-based Hierarchical (SMITH) Encoder for long-form document matching.
Our model contains several innovations to adapt self-attention models for longer
text input and shows promising results for long-form document matching in several
benchmark data sets. We released the model implementation and the pre-trained
model checkpoints in this repository.

## Pre-trained model checkpoint

We uploaded the pre-trained model checkpoint of the SMITH model to Google Cloud
Storage. You can download the model checkpoint following the instruction [here](http://storage.googleapis.com/gresearch/smith_gwikimatch/README.md).

## Usage

### Dependencies and Setup

The main dependencies include the following packages:

* python 3.7
* tensorflow 1.15
* nltk 3.5+
* tqdm 4.50.1+
* numpy 1.13.3+
* tf_slim 1.1.0

Here are the instructions on how to set up the environment (tested on Debian
Linux). After you download or clone our code, run the following commands
to set up the python3.7 virtual environment and install the required dependencies
with pip.

```
# ***From the folder google-research/***
sudo apt install python3.7 python3-venv python3.7-venv
python3.7 -m venv py37-venv
. py37-venv/bin/activate

pip install -r smith/requirements.txt
```

We used the sentence tokenizer in nltk to get the sentence boundary information
during data preprocessing. So you can run the following command to install the
nltk data resource:

```
python3

>>> import nltk
>>> nltk.download('punkt')
```

Next we need to set up protocol buffer. We used protocol buffer to define our model
configurations (experiment_config.proto) and the input raw text data containing
document pairs (wiki_doc_pair.proto). You can refer to this [page](https://grpc.io/docs/protoc-installation/)
on instructions of protocol buffer compiler installation. To install the latest
release of the protocol compiler from pre-compiled binaries, run the following
commands:

```
PB_REL="https://github.com/protocolbuffers/protobuf/releases"
curl -LO $PB_REL/download/v3.13.0/protoc-3.13.0-linux-x86_64.zip
unzip protoc-3.13.0-linux-x86_64.zip -d $HOME/.local
export PATH="$PATH:$HOME/.local/bin"
```

Once the protocol buffer compiler installation is finished, you can type protoc
in your terminal to see whether the installation is successful.

Next run the following commands to generate the python files given the proto
definition files.

```
# ***From the folder google-research/***
protoc smith/wiki_doc_pair.proto --python_out=.
protoc smith/experiment_config.proto --python_out=.
```

By running these commands, you set both of the input proto file path and the output
python file path as smith, which is the root path of our released code. For more
details on protoc and protocal buffer, you can refer to this [tutorial](https://developers.google.com/protocol-buffers/docs/pythontutorial).

Next run the following commands to make sure all the python test cases can
pass in the environment setup by you.

```
# ***From the folder google-research/***
python -m smith.loss_fns_test
python -m smith.metric_fns_test
python -m smith.modeling_test
python -m smith.preprocessing_smith_test
```

If all test cases passed, you have successfully set up the environment.

### Data Preprocessing

Before starting model training, you need to do data preprocessing. Let's take
the [GWikiMatch](https://github.com/google-research/google-research/tree/master/gwikimatch) data
as an example. Download GWikiMatch data following the data
[README file](https://github.com/google-research/google-research/tree/master/gwikimatch).
Then download the BERT base checkpoint with the BERT vocabulary file by running:

```
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```

After downloading the GWikiMatch data and BERT checkpoint, you can run the
following command to start the data preprocessing of SMITH model:

```
# ***From the folder google-research/***
python -m smith.preprocessing_smith \
--input_file=/path/to/input_file \
--output_file=/path/to/output_file \
--vocab_file=/path/to/bert/checkpoint/uncased_L-12_H-768_A-12/vocab.txt \
--max_sent_length_by_word=32 \
--max_doc_length_by_sentence=64 \
--max_predictions_per_seq=0 \
--add_masks_lm=false
```

If you put GWikiMatch data and BERT vocabulary file into path /tmp/data, the
command is:

```
# ***From the folder google-research/***
export DATA_DIR=/tmp/data/
python -m smith.preprocessing_smith \
--input_file=${DATA_DIR}gwikimatch_v2_human_neg_1.eval_external_wdp.tfrecord \
--output_file=${DATA_DIR}gwikimatch_v2_human_neg_1.eval_external_wdp_smith_32_64_false.tfrecord \
--vocab_file=${DATA_DIR}uncased_L-12_H-768_A-12/vocab.txt \
--max_sent_length_by_word=32 \
--max_doc_length_by_sentence=64 \
--max_predictions_per_seq=0 \
--add_masks_lm=false
```

You can finish the data preprocessing of the train/eval/test partition of
GWikiMatch data with these commands.

### Fine-tuning with SMITH

We use a model config file to maintain the settings of different models and
data during model training and evaluation. The proto buffer definition of the
model configuration is experiment_config.proto. All the example config files
can be found in the [config]() folder. Here we show how to run fine-tuning
experiments with the SMITH model. We take SMITH-Short and SMITH-WP+SP presented
in our [paper](https://research.google/pubs/pub49617/) as the example. Note that
SMITH-WP+SP and SMITH-Short are two model variations. SMITH-WP+SP is the best
model variation pretrained by ourselves.

SMITH-Short is the model variation where we load the BERT base checkpoint and
then fine-tune the model with only the document matching loss. The example
config file for SMITH-Short is as follows:

```
encoder_config {
  model_name: "smith_dual_encoder"
  init_checkpoint: "/tmp/data/uncased_L-12_H-768_A-12/bert_model.ckpt"
  bert_config_file: "/tmp/data/config/bert_config.json"
  doc_bert_config_file: "/tmp/data/config/doc_bert_3l_768h_config.json"
  vocab_file: "/tmp/data/uncased_L-12_H-768_A-12/vocab.txt"
  max_seq_length: 32
  max_predictions_per_seq: 5
  max_sent_length_by_word: 32
  max_doc_length_by_sentence: 64
  loop_sent_number_per_doc: 8
  sent_bert_trainable: true
  max_masked_sent_per_doc: 0
  use_masked_sentence_lm_loss: false
  num_labels: 2
  doc_rep_combine_mode: "normal"
  doc_rep_combine_attention_size: 256
}
train_eval_config {
  input_file_for_train: "/tmp/data/gwikimatch_v2_human_neg_1.train.smith_msenl_32_mdl_64_lm_false.tfrecord"
  input_file_for_eval: "/tmp/data/gwikimatch_v2_human_neg_1.eval_external_wdp_smith_32_64_false.tfrecord"
  train_batch_size: 32
  eval_batch_size: 32
  predict_batch_size: 32
  max_eval_steps: 54
  save_checkpoints_steps: 10
  iterations_per_loop: 10
  eval_with_eval_data: true
  neg_to_pos_example_ratio: 1.0
}
loss_config {
  similarity_score_amplifier: 6.0
}
```

Note that you need to replace the path '/tmp/data' to the corresponding path in
your environment. After you prepare the config file, you can
start the model fine-tuning of SMITH-Short with the following command:

```
# ***From the folder google-research/***
export DATA_DIR=/tmp/data/
python -m smith.run_smith \
--dual_encoder_config_file=${DATA_DIR}config/dual_encoder_config.smith_short.32.8.pbtxt \
--output_dir=${DATA_DIR}res/gwm_smith_short_32_8/ \
--train_mode=finetune \
--num_train_steps=10000 \
--num_warmup_steps=1000 \
--schedule=train
```

For SMITH-WP+SP, which is the SMITH model pre-trained with both masked word
prediction loss and masked sentence block prediction loss on the pre-training
collection and then fine-tuned with document matching loss, the example config
file is as follows:

```
encoder_config {
  model_name: "smith_dual_encoder"
  init_checkpoint: "/tmp/data/smith_pretrain_model_ckpts/smith_wsp/model.ckpt-400000"
  bert_config_file: "/tmp/data/config/sent_bert_4l_config.json"
  doc_bert_config_file: "/tmp/data/config/doc_bert_3l_256h_config.json"
  vocab_file: "/tmp/data/uncased_L-12_H-768_A-12/vocab.txt"
  max_seq_length: 32
  max_predictions_per_seq: 5
  max_sent_length_by_word: 32
  max_doc_length_by_sentence: 64
  loop_sent_number_per_doc: 48
  sent_bert_trainable: true
  max_masked_sent_per_doc: 0
  use_masked_sentence_lm_loss: false
  num_labels: 2
  doc_rep_combine_mode: "normal"
  doc_rep_combine_attention_size: 256
}
train_eval_config {
  input_file_for_train: "/tmp/data/gwikimatch_v2_human_neg_1.train.smith_msenl_32_mdl_64_lm_false.tfrecord"
  input_file_for_eval: "/tmp/data/gwikimatch_v2_human_neg_1.eval_external_wdp_smith_32_64_false.tfrecord"
  train_batch_size: 32
  eval_batch_size: 32
  predict_batch_size: 32
  max_eval_steps: 54
  save_checkpoints_steps: 10
  iterations_per_loop: 10
  eval_with_eval_data: true
  neg_to_pos_example_ratio: 1.0
}
loss_config {
  similarity_score_amplifier: 6.0
}
```

Note that you need to download the pretrained SMITH checkpoint for this
experimental run. After you prepare the config file, you can start the model
fine-tuning of SMITH-WP+SP with the following command:

```
# ***From the folder google-research/***
export DATA_DIR=/tmp/data/
python -m smith.run_smith \
--dual_encoder_config_file=${DATA_DIR}config/dual_encoder_config.smith_wsp.32.48.pbtxt \
--output_dir=${DATA_DIR}res/gwm_smith_wsp_32_48/ \
--train_mode=finetune \
--num_train_steps=10000 \
--num_warmup_steps=1000 \
--schedule=train
```

### Model pre-training

Our released code also supports SMITH model pre-training with masked word LM
loss and masked sentence block LM loss described in our [paper](https://arxiv.org/abs/2004.12297).
In practice, you can use any plain text data (e.g. Wikipedia documents or other
textual datasets) for model pre-training. To do model pre-training with our code
base, you can set the FLAGS train_mode as 'pretrain' and set
'use_masked_sentence_lm_loss' in the model config file as true if you want to
use both masked word LM loss and masked sentence LM loss. You also need to
add LM related masks into the data during data preprocessing similar to BERT.
Our data preprocessing script supports adding LM masks into the data. Note that
the original data preprocessing pipeline was developed in C++ and distributed
data processing for better scalability on larger pre-training data sets.
We won't release the C++ version of data preprocessing code. But our released
Python data preprocessing scripts can also generate the same Tensorflow
examples for model pre-training.

## Release Notes

- Initial release: 10/9/2020

## Disclaimer

This is not an officially supported Google product.

## Citing

If you extend or use this code/model checkpoint, please cite the following paper:

```
@inproceedings{yang2020beyond,
  title={Beyond 512 Tokens: Siamese Multi-depth Transformer-based Hierarchical Encoder for Long-Form Document Matching},
  author={Liu Yang and Mingyang Zhang and Cheng Li and Michael Bendersky and Marc Najork},
  booktitle={CIKM},
  year={2020}
}
```
