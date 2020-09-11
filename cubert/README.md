# CuBERT

## Introduction

This is a repository for code, models and data accompanying the ICML 2020 paper
[Learning and Evaluating Contextual Embedding of Source Code]
(https://proceedings.icml.cc/static/paper_files/icml/2020/5401-Paper.pdf).

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
the `source_code.py.test` file along with the CuBERT subword vocabulary
(**to be released**). It should produce the output as illustrated in the
`subtokenized_source_code.py.json` file. To obtain, token-ID sequences for use
with TensorFlow models, the `decode_list` logic from
`code_to_subtokenized_sentences.py` can be skipped.

## The Multi-Headed Pointer Model

The `finetune_varmisuse_pointer_lib.py` file provides an implementation of the
multi-headed pointer model described in [Neural Program Repair by Jointly Learning to Localize and Repair]
(https://openreview.net/pdf?id=ByloJ20qtm) on top of the pre-trained CuBERT
model. The `model_fn_builder` function should be integrated into an appropriate
fine-tuning script along the lines of the [fine-tuning script of the BERT model]
(https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_classifier.py#L847).
