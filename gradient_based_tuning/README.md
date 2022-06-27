# Simple and Effective Gradient-Based Tuning of Sequence-to-Sequence Models

This repository contains the code used in the paper "[Simple and Effective Gradient-Based Tuning of Sequence-to-Sequence Models](https://openreview.net/forum?id=RBTUKLfQ_pc)"
by Jared Lichtarge, Chris Alberti, and Shankar Kumar, to be presented at AutoML 2022.


We describe a simple approach to tuning the hyper-parameters during training by
updating them with the gradient on the tuning-loss. This code implements this
approach in JAX. The code allows for the easy specification of any subset of
hyper-parameters to be learned during training. Specifically, it trains
Transformer models on translation data from WMT.

## Dependencies

All dependencies are listed in requirements.txt. Models are implemented in flax.
The WMT data is sourced from Tensorflow Datasets (https://www.tensorflow.org/datasets/api_docs/python/tfds).

### Python Environment
We suggest installing the library in a vitual environment as our code requires older versions of libraries. We recommend creating a virtualenv (or conda).

To install libraries using pip, run: \
`pip3 install -r requirements.txt`

## Files

The main runner is train.py. See example below.

## Example

`python -m train.py \
--eval_dataset_path=/data/lt-en_dev.tfr \
--guidance_dataset_path=/data/lt-en_guide_1percent.tfr \
--guided_hparam_types=learning_rate \
--model_dir=/models/ \
--save_checkpoints=True \
--train_with_guided_parameters=1 \
--training_dataset=wmt_lt_en \
--training_dataset_path=/data/lt-en_train_99percent.tfr \
--vocab_path=/data/lt-en.32k.vocab`



If you use this code, please cite the paper:

```bibtex
@inproceedings{
lichtarge2022simple,
title={Simple and Effective Gradient-Based Tuning of Sequence-to-Sequence Models},
author={Jared Lichtarge and Chris Alberti and Shankar Kumar},
booktitle={First Conference on Automated Machine Learning (Late-Breaking Workshop)},
year={2022},
url={https://openreview.net/forum?id=RBTUKLfQ_pc}
}
```
