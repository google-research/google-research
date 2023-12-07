# Screen2Words: Automatic Mobile UI Summarization with Multimodal Learning
This repository contains the model code and the experimental framework for "Screen2Words: Automatic Mobile UI Summarization with Multimodal Learning" by Bryan Wang, Gang Li, Xin Zhou, Zhourong Chen, Tovi Grossman and Yang Li, which is conditionally accepted in UIST 2021.

# Data
The 112,085 screen summaries we collected for model training and evaluation can be downloaded from https://github.com/google-research-datasets/screen2words

# Implementation
The screen2words models are implemented based on Transformer implementation in [TensorFlow Model Garden](https://github.com/tensorflow/models).

# Setup

### Python Virtual environment

To checkout the virtual environment of python, call:
```shell
source $HOME/.venv/bin/activate # or any other path to venv
```

# Tensorflow with CUDA / GPU

Install Tensorflow with Nvidia GPU support
https://stackoverflow.com/a/76836703/5164462

### Sources
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
src/create_tf_example_main.py \
--task=CREATE_VOCAB \
--dataset_paths=/.../RICO/unique_uis/combined/ \
--json_file_path=/.../screen2words/screen_summaries.csv \
--output_vocab_path=/tmp/word_vocab.txt \
--word_vocab_path=/tmp/word_vocab.txt-00000-of-00001 \
--output_tfexample_path=/tmp/tf_example
```
