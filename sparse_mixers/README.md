# Sparse Mixers: Combining MoE and Mixing to build a more efficient BERT


## Introduction

Sparse Mixer combines the capacity of sparsely gated Mixture-of-Experts (MoE)
with the speed and stability of linear, mixing transformations to form an
efficient encoder architecture.

Sparse Mixer slightly outperforms BERT on GLUE and SuperGLUE, but more
importantly runs >50% faster. We also include a faster variant, prosaically
named Fast Sparse Mixer, that very marginally underperforms (<0.2%) BERT on
SuperGLUE, but trains and runs nearly twice as fast.

This repo contains the model libraries and model training code. Models are
pre-trained on the [C4](https://www.tensorflow.org/datasets/catalog/c4) dataset
and fine-tuned on [GLUE](https://gluebenchmark.com/) and
[SuperGLUE](https://super.gluebenchmark.com/).

## Installation

This codebase requires Python 3. To download the code and install dependencies,
you can run a command like the following (ideally in a fresh Python environment,
e.g. from [virtualenv](https://pypi.org/project/virtualenv/)):

```
svn export https://github.com/google-research/google-research/trunk/sparse_mixers
pip install -r sparse_mixers/requirements.txt

# If using Google Cloud TPUs:
pip install cloud-tpu-client
```

Unit tests can be run via:

```
python3 -m unittest discover -s sparse_mixers -p '*_test.py'
```

When running the unit tests and all python commands mentioned later, the current
working directory must be the *parent* folder of the `sparse_mixers` folder.

## How to pre-train or fine-tune Sparse Mixers

First,
[download](https://storage.googleapis.com/gresearch/sparse_mixers/vocab/c4_bpe_sentencepiece.model)
the SentencePiece vocab model. You can then train through the command line
interface:

```
python3 -m sparse_mixers.main --workdir=$YOUR_WORKDIR --vocab_filepath=$VOCAB_FILEPATH --config=$CONFIG
```

Here,

*   `YOUR_WORKDIR` is the directory where model checkpoints and metrics will be
    written.
*   `VOCAB_FILEPATH` is the SentencePiece vocab model that you downloaded.
*   `CONFIG` is either `sparse_mixers/configs/pretraining.py` (for pre-training)
    or `sparse_mixers/configs/classification.py` (for fine-tuning); all model,
    data, pretrained checkpoints (see below) and training flags are configured
    in the respective config files.

You can also [prepare your own](https://github.com/google/sentencepiece) vocab
model, but then you will not be able to use the below pre-trained models (unless
you manually clear the pre-trained model embedding layer weights after loading
the checkpoint).

## Pre-trained models

Pre-trained models will be made available soon.


