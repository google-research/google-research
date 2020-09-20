# CuBERT

## Introduction

This is a repository for code, models and data accompanying the ICML 2020 paper
[Learning and Evaluating Contextual Embedding of Source Code](https://proceedings.icml.cc/static/paper_files/icml/2020/5401-Paper.pdf).

**The model checkpoints and datasets will be linked from this README within
the next few weeks.**

If you use the code, models or data released through this repository, please
cite the following paper:
```
@inproceedings{cubert,
author    = {Aditya Kanade and
             Petros Maniatis and
             Gogul Balakrishnan and
             Kensen Shi},
title     = {Learning and evaluating contextual embedding of source code},
booktitle = {Proceedings of the 37th International Conference on Machine Learning,
               {ICML} 2020, 12-18 July 2020},
series    = {Proceedings of Machine Learning Research},
publisher = {{PMLR}},
year      = {2020},
```

## The CuBERT Tokenizer

The CuBERT tokenizer for Python is implemented in `python_tokenizer.py`, as
a subclass of a language-agnostic tokenization framework in
`cubert_tokenizer.py`. `unified_tokenizer.py` contains useful utilities for
language-agnostic tokenization,
which can be extended along the lines of the Python tokenizer for other
languages. We show one other instance, for Java, in `java_tokenizer.py`,
although the original CuBERT benchmark is only about Python code.

The code within the `code_to_subtokenized_sentences.py` script can be used for
converting Python code (in fact, any language for which there's a subclass of
`CuBertTokenizer`) into CuBERT sentences. This script can be evaluated on
the `source_code.py.test` file along with a CuBERT subword vocabulary. It should
produce output similar to that illustrated in the
`subtokenized_source_code.py.json` file. To obtain token-ID sequences for use
with TensorFlow models, the `decode_list` logic from
`code_to_subtokenized_sentences.py` can be skipped.

## The Multi-Headed Pointer Model

The `finetune_varmisuse_pointer_lib.py` file provides an implementation of the
multi-headed pointer model described in [Neural Program Repair by Jointly Learning to Localize and Repair](https://openreview.net/pdf?id=ByloJ20qtm) on top of the pre-trained CuBERT
model. The `model_fn_builder` function should be integrated into an appropriate
fine-tuning script along the lines of the [fine-tuning script of the BERT model](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_classifier.py#L847).

## Pre-trained Datasets and Models

We provide the following files, all stored in Google Cloud Storage. We give
links to each file or directory (via the Cloud Storage UI), as well as URIs for the
corresponding dataset to be used via the [command-line interface](https://cloud.google.com/storage/docs/gsutil).

For each language we provide the following:

1. Manifest of files used during pre-training. This contains the precise specification of pre-training source files, which can be used with BigQuery or the GitHub API (see below for a sample query for BigQuery). The data are stored as sharded text files. Each text line contains a JSON-formatted object.
      * `repository`: string, the name of a GitHub repository.
      * `filepath`: string, the path from the repository root to the file mentioned.
      * `license`: string, (one of 'apache-2.0', 'lgpl-2.1', 'epl-1.0', 'isc', 'bsd-3-clause', 'bsd-2-clause', 'mit', 'gpl-2.0', 'cc0-1.0', 'lgpl-3.0', 'mpl-2.0', 'unlicense', 'gpl-3.0'); the license under which the file’s repository was released on GitHub.
      * `id`: string, a unique identifier under which the file’s content is hosted in BigQuery’s public GitHub repository.
      * `url`: string, a URL by which the GitHub API uniquely identifies the content.

1. Vocabulary file.
Used to encode pre-training and fine-tuning examples. It is extracted from the files pointed to by the Manifest of files. It is stored as a single text file, holding one quoted token per line, as per the format produced by [`tensor2tensor`'s `SubwordTextEncoder`](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder.py).

1. Pre-trained model checkpoint. These are as stored by BERT's [`run_pretraining.py`](https://github.com/google-research/bert/blob/master/run_pretraining.py). Although we used a modified version of the pre-training code to use the CuBERT tokenizer (see above), the models still have the BERT architecture and are stored in a compatible way. The actual BERT configuration is BERT-Large:
      * "attention_probs_dropout_prob": 0.1,
      * "hidden_act": "gelu",
      * "hidden_dropout_prob": 0.1,
      * "hidden_size": 1024,
      * "initializer_range": 0.02,
      * "intermediate_size": 4096,
      * "num_attention_heads": 16,
      * "num_hidden_layers": 24,
      * "type_vocab_size": 2,
      * "vocab_size": *corresponding vocabulary size*,
      * "max_position_embeddings": *corresponding sequence length*

To retrieve a pre-training file, given its `id`, you can use the following [BigQuery query](https://console.cloud.google.com/bigquery):
```
select files.repo_name, files.path, files.ref, contents.content
from `bigquery-public-data.github_repos.files` as files,
     `bigquery-public-data.github_repos.contents` as contents
where contents.id = files.id and
      contents.id = <id>;
```

At this time, we release the following pre-trained datasets and models:

1. Python, deduplicated after files similar to [ETH Py150 Open](https://github.com/google-research-datasets/eth_py150_open) were removed. BigQuery snapshot as of June 21 2020.
    * Manifest: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/github_python_minus_ethpy150open_deduplicated_manifest)
        [`gs://cubert/20200621_Python/github_python_minus_ethpy150open_deduplicated_manifest`].
    * Vocabulary: [[UI]](https://console.cloud.google.com/storage/browser/_details/cubert/20200621_Python/github_python_minus_ethpy150open_deduplicated_vocabulary.txt)
        [`gs://cubert/20200621_Python/github_python_minus_ethpy150open_deduplicated_vocabulary.txt`].
    * Model checkpoint for length 512, 1 epoch: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/pre_trained_model__epochs_1__length_512)
        [`gs://cubert/20200621_Python/pre_trained_model__epochs_1__length_512`].

1. Java, **not deduplicated**, BigQuery snapshot as of September 13 2020.
    * Manifest: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200913_Java/github_java_manifest)
        [`gs://cubert/20200913_Java/github_java_manifest`].
    * Vocabulary: [[UI]](https://console.cloud.google.com/storage/browser/_details/cubert/20200913_Java/github_java_vocabulary.txt)
        [`gs://cubert/20200913_Java/github_java_vocabulary.txt`].
    * Model checkpoint for length 1024, 0.03 epochs: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200913_Java/pre_trained_model__epochs_0.03__length_1024)
        [`gs://cubert/20200913_Java/pre_trained_model__epochs_0.03__length_1024`].

## Benchmarks and Fine-Tuned Models

_Coming soon._

