# Screen2Words: Automatic Mobile UI Summarization with Multimodal Learning
This repository contains the model code and the experimental framework for "Screen2Words: Automatic Mobile UI Summarization with Multimodal Learning" by Bryan Wang, Gang Li, Xin Zhou, Zhourong Chen, Tovi Grossman and Yang Li, which is conditionally accepted in UIST 2021.

# Data
The 112,085 screen summaries we collected for model training and evaluation can be downloaded from https://github.com/google-research-datasets/screen2words

# Implementation
The screen2words models are implemented based on Transformer implementation in [TensorFlow Model Garden](https://github.com/tensorflow/models).

# Setup

Download following datasets:

- RICO's UI Screenshots and View Hierarchies from http://www.interactionmining.org/rico.html
- screen2words dataset from https://github.com/google-research-datasets/screen2words

Download a pre-trained GloVe model (e.g. Common Crawl) from https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors

Prepare nltk
```bash
python3
>>> import nltk
>>> nltk.download()
>>> book
```

Pre-process RICO Data with:
```bash
screen2words/create_tf_example_main.py \
--dataset_paths=/.../RICO/unique_uis/combined/ \
--json_file_path=/.../screen2words/screen_summaries.csv \
--word_vocab_path=/tmp/word_vocab.txt
```