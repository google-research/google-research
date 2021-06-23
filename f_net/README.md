# FNet: Mixing Tokens with Fourier Transforms

by James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santiago Ontanon.


## Introduction

[FNet](https://arxiv.org/abs/2105.03824) is a highly efficient Transformer-like
encoder architecture, wherein the self-attention sublayers have been wholly
replaced by standard, unparameterized Fourier Transforms.

This repo contains the models and code required to reproduce the results in the
[paper](https://arxiv.org/abs/2105.03824).

Models are pre-trained on the
[C4](https://www.tensorflow.org/datasets/catalog/c4) dataset and fine-tuned on
[GLUE](https://gluebenchmark.com/). The code is implemented in
[Jax](https://jax.readthedocs.io)/[Flax](http://flax.readthedocs.io).


## Installation

To download the code and install dependencies,
you can run a command like the following (ideally in a fresh Python environment,
e.g. from [virtualenv](https://pypi.org/project/virtualenv/)):

```
svn export https://github.com/google-research/google-research/trunk/f_net
pip install -r f_net/requirements.txt

# If using Google Cloud TPUs:
pip install cloud-tpu-client
```

Unit tests can be run via:

```
python3 -m unittest discover -s f_net -p '*_test.py'
```

When running the unit tests and all python commands mentioned later, the current
working directory must be the *parent* folder of the `f_net` folder.


## How to pre-train or fine-tune FNet

First,
[download](https://storage.googleapis.com/gresearch/f_net/vocab/c4_bpe_sentencepiece.model)
the SentencePiece vocab model. You can then train through the command line
interface:

```
python3 -m f_net.main --workdir=$YOUR_WORKDIR --vocab_filepath=$VOCAB_FILEPATH --config=$CONFIG
```

Here,

*   `YOUR_WORKDIR` is the directory where model checkpoints and metrics will be
    written.
*   `VOCAB_FILEPATH` is the SentencePiece vocab model that you downloaded.
*   `CONFIG` is either `f_net/configs/pretraining.py` (for pre-training) or
    `f_net/configs/classification.py` (for fine-tuning); all model, data,
    pretrained checkpoints (see below) and training flags are configured in the
    respective config files.
    

You can also [prepare your own](https://github.com/google/sentencepiece) vocab
model, but then you will not be able to use the below pre-trained models (unless
you manually clear the pre-trained model embedding layer weights after loading
the checkpoint).



## Pre-trained models

Pre-trained model checkpoints can be downloaded and then used in training by
updating the relevant training config file. All pre-trained models use
`max_seq_length=512` and a `type_vocab_size=4`, which allows for up to 4 input
segments (the number of input segments is automatically configured for most
tasks).


### Base models

Base models use

```
config.d_emb = 768
config.d_model = 768
config.d_ff = 3072
config.num_heads = 12
config.num_layers = 12
```

The following Base model pre-trained checkpoints are available (~1 GB file
sizes):

*   [ModelArchitecture.F_NET](https://storage.googleapis.com/gresearch/f_net/checkpoints/base/f_net_checkpoint)
*   [ModelArchitecture.LINEAR](https://storage.googleapis.com/gresearch/f_net/checkpoints/base/linear_checkpoint)
*   [ModelArchitecture.BERT](https://storage.googleapis.com/gresearch/f_net/checkpoints/base/bert_checkpoint)
*   [ModelArchitecture.FF_ONLY](https://storage.googleapis.com/gresearch/f_net/checkpoints/base/ff_only_checkpoint)
*   [ModelArchitecture.RANDOM](https://storage.googleapis.com/gresearch/f_net/checkpoints/base/random_checkpoint)


### Large models

Large models use

```
config.d_emb = 1024
config.d_model = 1024
config.d_ff = 4096
config.num_heads = 16
config.num_layers = 24
```

The following Large model pre-trained checkpoints are available (3-4 GB file
sizes):

*   [ModelArchitecture.F_NET](https://storage.googleapis.com/gresearch/f_net/checkpoints/large/f_net_checkpoint)
*   [ModelArchitecture.LINEAR](https://storage.googleapis.com/gresearch/f_net/checkpoints/large/linear_checkpoint)
*   [ModelArchitecture.BERT](https://storage.googleapis.com/gresearch/f_net/checkpoints/large/bert_checkpoint)



## Bibtex

```bibtex
@article{lee2021fnet,
  title={FNet: Mixing Tokens with Fourier Transforms},
  author={Lee-Thorp, James and Ainslie, Joshua and Eckstein, Ilya and Ontanon, Santiago},
  journal={arXiv preprint arXiv:2105.03824},
  year={2021}
}
```

**This is not an official Google product.**
