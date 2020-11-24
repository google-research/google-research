# CuBERT


## Update 2020/11/16: Pre-trained Java Model with Code Comments

We are releasing a Java pre-training corpus and pre-trained model. This model was pre-trained on all Java content, including comments.

* Java, deduplicated, with code comments, BigQuery snapshot as of October 18, 13, 2020.
    * Manifest: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20201018_Java_Deduplicated/github_java_manifest)
        [`gs://cubert/20201018_Java_Deduplicated/github_java_manifest`].
    * Vocabulary: [[UI]](https://console.cloud.google.com/storage/browser/_details/cubert/20201018_Java_Deduplicated/github_java_vocabulary.txt)
        [`gs://cubert/20201018_Java_Deduplicated/github_java_vocabulary.txt`].
    * Model checkpoint for length 1024, 1 epoch: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20201018_Java_Deduplicated/pre_trained_model_deduplicated__epochs_1__length_1024)
        [`gs://cubert/20201018_Java_Deduplicated/pre_trained_model_deduplicated__epochs_1__length_1024`].


## Update 2020/09/29: Pre-trained Java Model

We are releasing a Java pre-training corpus and pre-trained model. This model was not pre-trained on comments, but an expanded model including Javadoc and regular comments is upcoming.

* Java, deduplicated, no code comments, BigQuery snapshot as of September 13, 2020.
    * Manifest: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200913_Java_Deduplicated/github_java_manifest)
        [`gs://cubert/20200913_Java_Deduplicated/github_java_manifest`].
    * Vocabulary: [[UI]](https://console.cloud.google.com/storage/browser/_details/cubert/20200913_Java_Deduplicated/github_java_vocabulary.txt)
        [`gs://cubert/20200913_Java_Deduplicated/github_java_vocabulary.txt`].
    * Model checkpoint for length 1024, 1 epoch: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200913_Java_Deduplicated/pre_trained_model_deduplicated__epochs_1__length_1024)
        [`gs://cubert/20200913_Java_Deduplicated/pre_trained_model_deduplicated__epochs_1__length_1024`].


## Introduction

This is a repository for code, models and data accompanying the ICML 2020 paper
[Learning and Evaluating Contextual Embedding of Source Code](https://proceedings.icml.cc/static/paper_files/icml/2020/5401-Paper.pdf). In addition to the Python artifacts described in the paper, we are also
releasing the pre-training corpus and CuBERT models for other languages.

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
}
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

## Pre-trained Models and Pre-training Corpora

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

At this time, we release the following pre-trained model and pre-training corpus. Look in the updates, below, for other releases.

* Python, deduplicated after files similar to [ETH Py150 Open](https://github.com/google-research-datasets/eth_py150_open) were removed. BigQuery snapshot as of June 21, 2020.
    * Manifest: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/github_python_minus_ethpy150open_deduplicated_manifest)
        [`gs://cubert/20200621_Python/github_python_minus_ethpy150open_deduplicated_manifest`].
    * Vocabulary: [[UI]](https://console.cloud.google.com/storage/browser/_details/cubert/20200621_Python/github_python_minus_ethpy150open_deduplicated_vocabulary.txt)
        [`gs://cubert/20200621_Python/github_python_minus_ethpy150open_deduplicated_vocabulary.txt`].
    * Model checkpoint for length 512, 1 epoch: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/pre_trained_model__epochs_1__length_512)
        [`gs://cubert/20200621_Python/pre_trained_model__epochs_1__length_512`].
    * Model checkpoint for length 512, 2 epochs: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/pre_trained_model__epochs_2__length_512)
        [`gs://cubert/20200621_Python/pre_trained_model__epochs_2__length_512`].


## Benchmarks and Fine-Tuned Models

Here we describe the 6 Python benchmarks we created. All 6 benchmarks were derived from [ETH Py150 Open](https://github.com/google-research-datasets/eth_py150_open). All examples are stored as sharded text files. Each text line corresponds to a separate example encoded as a JSON object. For each dataset, we release separate training/validation/testing splits along the same boundaries that ETH Py150 Open splits its files to the corresponding splits. The fine-tuned models are the checkpoints of each model with the highest validation accuracy.

1. **Function-docstring classification**. Combinations of functions with their correct or incorrect documentation string, used to train a classifier that can tell which pairs go together. The JSON fields are:
     * `function`: string, the source code of a function as text
     * `docstring`: string, the documentation string for that function
     * `label`: string, one of (“Incorrect”, “Correct”), the label of the example.
     * `info`: string, an unformatted description of how the example was constructed, including the source dataset (always “ETHPy150Open”), the repository and filepath, the function name and, for “Incorrect” examples, the function whose docstring was substituted.
1. **Exception classification**. Combinations of functions where one exception type has been masked, along with a label indicating the masked exception type. The JSON fields are:
     * `function`: string, the source code of a function as text, in which one exception type has been replaced with the special token “__HOLE__”
     * `label`: string, one of (`ValueError`, `KeyError`, `AttributeError`, `TypeError`, `OSError`, `IOError`, `ImportError`, `IndexError`, `DoesNotExist`, `KeyboardInterrupt`, `StopIteration`, `AssertionError`, `SystemExit`, `RuntimeError`, `HTTPError`, `UnicodeDecodeError`, `NotImplementedError`, `ValidationError`, `ObjectDoesNotExist`, `NameError`, `None`), the masked exception type. Note that `None` never occurs in the data and will be removed in a future release.
     * `info`: string, an unformatted description of how the example was constructed, including the source dataset (always “ETHPy150Open”), the repository and filepath, and the fully-qualified function name.
1. **Variable-misuse classification**. Combinations of functions where one use of a variable may have been replaced with another variable defined in the same context, along with a label indicating if this bug-injection has occurred. The JSON fields are:
     * `function`: string, the source code of a function as text.
     * `label`: string, one of (“Correct”, “Variable misuse”) indicating if this is a buggy or bug-free example.
     * `info`: string, an unformatted description of how the example was constructed, including the source dataset (always “ETHPy150Open”), the repository and filepath, the function, and whether the example is bugfree (marked “original”) or the variable substitution that has occurred (e.g., “correct_variable” → “incorrect_variable”).
1. **Swapped-operand classification**. Combinations of functions where one use binary operator’s arguments have been swapped, to create a buggy example, or left undisturbed, along with a label indicating if this bug-injection has occurred. The JSON fields are:
     * `function`: string, the source code of a function as text.
     * `label`: string, one of (“Correct”, “Swapped operands”) indicating if this is a buggy or bug-free example.
     * `info`: string, an unformatted description of how the example was constructed, including the source dataset (always “ETHPy150Open”), the repository and filepath, the function, and whether the example is bugfree (marked “original”) or the operand swap has occurred (e.g., “swapped operands of `not in`”).
1. **Wrong-binary-operator classification**. Combinations of functions where one binary operator has been swapped with another, to create a buggy example, or left undisturbed, along with a label indicating if this bug-injection has occurred. The JSON fields are:
     * `function`: string, the source code of a function as text.
     * `label`: string, one of (“Correct”, “Wrong binary operator”) indicating if this is a buggy or bug-free example.
     * `info`: string, an unformatted description of how the example was constructed, including the source dataset (always “ETHPy150Open”), the repository and filepath, the function, and whether the example is bugfree (marked “original”) or the operator replacement has occurred (e.g., “`==`-> `!=`”).
1. **Variable-misuse localization and repair**. Combinations of functions where one use of a variable may have been replaced with another variable defined in the same context, along with information that can be used to localize and repair the bug, as well as the location of the bug if such a bug exists. The JSON fields are:
     * `function`: a list of strings, the source code of a function, tokenized with the vocabulary from item b. Note that, unlike other task datasets, this dataset gives a tokenized function, rather than the code as a single string.
     * `target_mask`: a list of integers (0 or 1). If the integer at some position is 1, then the token at the corresponding position of the function token list is a correct repair for the introduced bug. If a variable has been split into multiple tokens, only the first subtoken is marked in this mask. If the example is bug-free, all integers are 0.
     * `error_location_mask`: a list of integers (0 or 1). If the integer at some position is 1, then there is a variable-misuse bug at the corresponding location of the tokenized function. In a bug-free example, the first integer is 1. There is exactly one integer set to 1 for all examples. If a variable has been split into multiple tokens, only the first subtoken is marked in this mask.
     * `candidate_mask`: a list of integers (0 or 1). If the integer at some position is 1, then the variable starting at that position in the tokenized function is a candidate to consider when repairing a bug. Candidates are all variables defined in the function parameters or via variable declarations in the function. If a variable has been split into multiple tokens, only the first subtoken is marked in this mask, for each candidate.
     * `provenance`: string, an unformatted description of how the example was constructed, including the source dataset (always “ETHPy150Open”), the repository and filepath, the function, and whether the example is bugfree (marked “original”) or the buggy/repair token positions and variables (e.g., “16/18 `kwargs` → `self`”). 16 is the position of the introduced error, 18 is the location of the repair.


We release the following file collections:

1. **Function-docstring classification**.
    * Dataset: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/function_docstring_datasets)
        [`gs://cubert/20200621_Python/function_docstring_datasets`].
    * Fine-tuned Model: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/function_docstring__epochs_20__pre_trained_epochs_1)
        [`gs://cubert/20200621_Python/function_docstring__epochs_20__pre_trained_epochs_1`].
1. **Exception classification**.
    * Dataset: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/exception_datasets)
        [`gs://cubert/20200621_Python/exception_datasets`].
    * Fine-tuned Model: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/exception__epochs_20__pre_trained_epochs_1)
        [`gs://cubert/20200621_Python/exception__epochs_20__pre_trained_epochs_1`].
1. **Variable-misuse classification**.
    * Dataset: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/variable_misuse_datasets)
        [`gs://cubert/20200621_Python/variable_misuse_datasets`].
    * Fine-tuned Model: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/variable_misuse__epochs_20__pre_trained_epochs_1)
        [`gs://cubert/20200621_Python/variable_misuse__epochs_20__pre_trained_epochs_1`].
1. **Swapped-operand classification**.
    * Dataset: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/swapped_operands_datasets)
        [`gs://cubert/20200621_Python/swapped_operands_datasets`].
    * Fine-tuned Model: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/swapped_operands__epochs_20__pre_trained_epochs_1)
        [`gs://cubert/20200621_Python/swapped_operands__epochs_20__pre_trained_epochs_1`].
1. **Wrong-binary-operator classification**.
    * Dataset: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/wrong_binary_operator_datasets)
        [`gs://cubert/20200621_Python/wrong_binary_operator_datasets`].
    * Fine-tuned Model: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/wrong_binary_operator__epochs_20__pre_trained_epochs_1)
        [`gs://cubert/20200621_Python/wrong_binary_operator__epochs_20__pre_trained_epochs_1`].
1. **Variable-misuse localization and repair**.
    * Dataset: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/variable_misuse_repair_datasets)
        [`gs://cubert/20200621_Python/variable_misuse_repair_datasets`].
    * Fine-tuned Model: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/variable_misuse_repair__epochs_20__pre_trained_epochs_1)
        [`gs://cubert/20200621_Python/variable_misuse_repair__epochs_20__pre_trained_epochs_1`].
