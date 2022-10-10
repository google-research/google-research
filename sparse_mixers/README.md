# Sparse Mixers: Combining MoE and Mixing to build a more efficient BERT

James Lee-Thorp and Joshua Ainslie.

## Introduction

[Sparse Mixer](https://arxiv.org/abs/2205.12399) combines the capacity of
sparsely gated Mixture-of-Experts (MoE) with the speed and stability of linear,
mixing transformations to form an efficient encoder architecture.

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

Pre-trained model checkpoints can be downloaded and then used in training by
updating the relevant training config file. All pre-trained models use
`max_seq_length=512` and a `type_vocab_size=4`, which allows for up to 4 input
segments (the number of input segments is automatically configured for most
tasks).

### Base models

[Sparse Mixer](https://storage.googleapis.com/gresearch/sparse_mixers/checkpoints/base/sparse_mixer/sm_base.zip).
The default config matches the Sparse Mixer Base model:

```
config.arch = 'linear'
config.d_emb = 512
config.d_model = 512
config.d_ff = 2048
config.num_heads = 8
config.num_layers = 14
config.num_attention_layers = 4
config.moe_layers = 4
config.num_experts = 16
```

[Fast Sparse Mixer](https://storage.googleapis.com/gresearch/sparse_mixers/checkpoints/base/fast_sparse_mixer/fsm_base.zip).
Same as Sparse Mixer Base but with:

```
config.train_capacity_factor = 0.5
config.eval_capacity_factor = 0.5
```

[BERT](https://storage.googleapis.com/gresearch/sparse_mixers/checkpoints/base/bert/replicated_checkpoint_990000).
BERT Base uses:

```
config.arch = 'bert'
config.d_emb = 768
config.d_model = 768
config.d_ff = 3072
config.num_heads = 12
config.num_layers = 12
config.moe_layers = 0
config.num_experts = 0
```

### Small models

[Sparse Mixer](https://storage.googleapis.com/gresearch/sparse_mixers/oss/checkpoints/small/sparse_mixer/sm_small.zip).
Sparse Mixer Small uses (8 layers instead of 14):

```
config.arch = 'linear'
config.d_emb = 512
config.d_model = 512
config.d_ff = 2048
config.num_heads = 8
config.num_layers = 8
config.num_attention_layers = 4
config.moe_layers = 4
config.num_experts = 16
```

[Fast Sparse Mixer](https://storage.googleapis.com/gresearch/sparse_mixers/oss/checkpoints/small/fast_sparse_mixer/fsm_small.zip).
Same as Sparse Mixer Small but with:

```
config.train_capacity_factor = 0.5
config.eval_capacity_factor = 0.5
```

## Bibtex

To cite this work, please use:

```
@article{lee2022sparse,
  title={Sparse Mixers: Combining MoE and Mixing to build a more efficient BERT},
  author={Lee-Thorp, James and Ainslie, Joshua},
  journal={arXiv preprint arXiv:2205.12399},
  year={2022}
}
```
